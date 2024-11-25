from dataclasses import dataclass
from typing import TYPE_CHECKING
import polars as pl
from datetime import datetime
from aligned.data_source.batch_data_source import CodableBatchDataSource
from aligned.feature_source import WritableFeatureSource
from aligned.request.retrival_request import RetrivalRequest
from aligned.schemas.feature import Feature
from aligned.sources.local import Deletable
import logging

from aligned.sources.vector_index import VectorIndex

if TYPE_CHECKING:
    from aligned.retrival_job import RetrivalJob


try:
    import lancedb
except ImportError:
    lancedb = None

logger = logging.getLogger(__name__)


@dataclass
class LanceDBConfig:

    path: str

    async def connect(self) -> 'lancedb.AsyncConnection':  # type: ignore
        assert lancedb is not None, '`lancedb` is not installed'
        return await lancedb.connect_async(self.path)

    async def connect_to_table(self, table: str) -> 'lancedb.AsyncTable':  # type: ignore
        conn = await self.connect()
        return await conn.open_table(table)

    def table(self, name: str) -> 'LanceDbTable':
        return LanceDbTable(table_name=name, config=self)


@dataclass
class LanceDbTable(VectorIndex, CodableBatchDataSource, WritableFeatureSource, Deletable):

    table_name: str
    config: LanceDBConfig

    _vector_index_name: str | None = None

    type_name = 'lancedb_table'

    def job_group_key(self) -> str:
        return self.config.path + self.table_name

    def vector_index_name(self) -> str | None:
        return self.table_name

    def as_vector_index(self, name: str) -> 'LanceDbTable':
        self._vector_index_name = name
        return self

    async def freshness(self, feature: Feature) -> datetime | None:
        from lancedb import AsyncTable

        try:
            lance_table: AsyncTable = await self.config.connect_to_table(self.table_name)
            table = await lance_table.query().select([feature.name]).to_arrow()
            df = pl.from_arrow(table)
            if isinstance(df, pl.Series):
                col = df
            else:
                col = df.get_column(feature.name)

            max_value = col.max()
            if max_value is not None:
                assert isinstance(max_value, datetime)

            return max_value
        except ValueError:
            logger.info(f"Unable to load freshness. Assumes that it do not exist for '{self.table_name}'")
            return None

    async def create(self, request: RetrivalRequest) -> None:
        from aligned.schemas.vector_storage import pyarrow_schema

        db = await self.config.connect()
        schema = pyarrow_schema(list(request.all_returned_features))

        await db.create_table(self.table_name, schema=schema)

    async def upsert(self, job: 'RetrivalJob', request: RetrivalRequest) -> None:
        import lancedb

        upsert_keys = list(request.entity_names)

        conn = lancedb.connect(self.config.path)
        table = conn.open_table(self.table_name)

        df = await job.to_polars()
        if df.is_empty():
            return

        arrow_table = df.to_arrow()

        # Is a bug when passing in an iterator
        # As lancedb trys to access the .iter() which do not always exist I guess
        (
            table.merge_insert(upsert_keys[0] if len(upsert_keys) == 1 else upsert_keys)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(arrow_table)
        )

    async def insert(self, job: 'RetrivalJob', request: RetrivalRequest) -> None:
        try:
            table = await self.config.connect_to_table(self.table_name)
        except ValueError:
            await self.create(request)
            table = await self.config.connect_to_table(self.table_name)

        df = await job.to_polars()
        if df.is_empty():
            return

        arrow_table = df.to_arrow()
        await table.add(arrow_table)

    async def overwrite(self, job: 'RetrivalJob', request: RetrivalRequest) -> None:
        await self.delete()
        await self.insert(job, request)

    async def delete(self) -> None:
        conn = await self.config.connect()
        await conn.drop_table(self.table_name)

    def all_data(self, request: RetrivalRequest, limit: int | None) -> 'RetrivalJob':
        from aligned.retrival_job import RetrivalJob

        async def load() -> pl.LazyFrame:
            table = await self.config.connect_to_table(self.table_name)
            query = table.query().select(request.all_returned_columns)
            if limit:
                query = query.limit(limit)

            df = pl.from_arrow(await table.query().to_arrow())
            if isinstance(df, pl.DataFrame):
                return df.lazy()
            else:
                return pl.DataFrame(df).lazy()

        return RetrivalJob.from_lazy_function(load, request)

    def nearest_n_to(
        self, data: 'RetrivalJob', number_of_records: int, request: RetrivalRequest
    ) -> 'RetrivalJob':
        from aligned.retrival_job import RetrivalJob

        async def load() -> pl.LazyFrame:
            def first_embedding(features: set[Feature]) -> Feature | None:
                for feature in features:
                    if feature.dtype.is_embedding:
                        return feature
                return None

            df = await data.to_polars()

            table = await self.config.connect_to_table(self.table_name)
            result: pl.DataFrame | None = None

            embedding = first_embedding(data.request_result.features)
            assert embedding, 'Expected to a least find one embedding in the input data'
            org_columns = df.columns
            df_cols = len(df.columns)

            for item in df.iter_rows(named=True):
                nearest = (
                    await table.query().nearest_to(item[embedding.name]).limit(number_of_records).to_arrow()
                )

                polars_df = pl.from_arrow(nearest)
                assert isinstance(polars_df, pl.DataFrame), f"Expected a data frame, was {type(polars_df)}"

                polars_df = polars_df.select(pl.exclude('_distance'))
                if df_cols > 1:
                    logger.info(f"Stacking {polars_df.columns} and {item.keys()}")
                    polars_df = polars_df.select(pl.exclude(org_columns)).hstack(
                        pl.DataFrame([item] * polars_df.height)
                        .select(org_columns)
                        .select(pl.exclude(embedding.name))
                    )

                if result is None:
                    result = polars_df
                else:
                    result = result.vstack(polars_df)

            if result is None:
                return pl.DataFrame().lazy()
            else:
                return result.lazy()

        return RetrivalJob.from_lazy_function(load, request)

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls, facts: 'RetrivalJob', requests: list[tuple['LanceDbTable', RetrivalRequest]]
    ) -> 'RetrivalJob':
        from aligned.retrival_job import RetrivalJob

        if len(requests) != 1:
            raise ValueError(f"Expected only one request. Got: {len(requests)}")

        source, request = requests[0]

        if len(request.entity_names) != 1:
            raise ValueError(
                f"Only supporting key value lookup for now, with one entity. Got: {len(request.entity_names)}"
            )

        async def load() -> pl.LazyFrame:
            entities = await facts.select(request.entity_names).to_polars()

            entity = entities.columns[0]
            joined_values = "'" + "', '".join(entities[entity].to_list()) + "'"
            filter = f"{entity} IN ({joined_values})"

            conn = await source.config.connect_to_table(source.table_name)
            arr = (conn.query().select(request.all_returned_columns).where(filter)).to_arrow()

            df = pl.from_arrow(arr)
            assert isinstance(df, pl.DataFrame)
            return df.lazy()

        return RetrivalJob.from_lazy_function(load, request)
