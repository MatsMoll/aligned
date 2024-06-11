from dataclasses import dataclass
from typing import TYPE_CHECKING
import polars as pl
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.feature_source import WritableFeatureSource
from aligned.request.retrival_request import RetrivalRequest
from aligned.schemas.feature import Feature
from aligned.sources.local import Deletable

from aligned.sources.vector_index import VectorIndex

if TYPE_CHECKING:
    from aligned.retrival_job import RetrivalJob


try:
    import lancedb
except ImportError:
    lancedb = None


@dataclass
class LanceDBConfig:

    path: str

    async def connect(self) -> 'lancedb.AsyncConnection':
        return await lancedb.connect_async(self.path)

    async def connect_to_table(self, table: str) -> 'lancedb.AsyncTable':
        conn = await self.connect()
        return await conn.open_table(table)

    def table(self, name: str) -> 'LanceDbTable':
        return LanceDbTable(table_name=name, config=self)


@dataclass
class LanceDbTable(VectorIndex, BatchDataSource, WritableFeatureSource, Deletable):

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

    async def insert(self, job: 'RetrivalJob', request: RetrivalRequest) -> None:
        table = await self.config.connect_to_table(self.table_name)
        df = (await job.to_polars()).to_arrow()
        await table.add(df)

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
        self, data: 'RetrivalJob', number_of_records: int, retrival_request: RetrivalRequest
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

            df_cols = len(df.columns)

            for item in df.iter_rows(named=True):
                nearest = (
                    await table.query().nearest_to(item[embedding.name]).limit(number_of_records).to_arrow()
                )

                polars_df = pl.from_arrow(nearest)
                assert isinstance(polars_df, pl.DataFrame), f"Expected a data frame, was {type(polars_df)}"

                polars_df = polars_df.select(pl.exclude('_distance'))
                if df_cols > 1:
                    polars_df = polars_df.hstack(pl.DataFrame(item).select(pl.exclude(embedding.name)))

                if result is None:
                    result = polars_df
                else:
                    result = result.vstack(polars_df)

            if result is None:
                return pl.DataFrame().lazy()
            else:
                return result.lazy()

        return RetrivalJob.from_lazy_function(load, retrival_request)

    @classmethod
    def multi_source_features_for(
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
