from typing import TYPE_CHECKING

import uuid

import polars as pl
from aligned.data_file import DataFileReference, upsert_on_column
from aligned.data_source.batch_data_source import BatchDataSource, CodableBatchDataSource
from aligned.feature_source import WritableFeatureSource
from aligned.retrival_job import RetrivalJob, RetrivalRequest
from aligned.schemas.feature import Feature
from aligned.sources.vector_index import VectorIndex

if TYPE_CHECKING:
    from aligned.schemas.feature_view import CompiledFeatureView


class InMemorySource(CodableBatchDataSource, DataFileReference, WritableFeatureSource, VectorIndex):

    type_name = 'in_mem_source'

    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data
        self.job_key = str(uuid.uuid4())
        self._vector_index_name = None

    def vector_index_name(self) -> str | None:
        return self._vector_index_name

    def job_group_key(self) -> str:
        return self.job_key

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return self.data.lazy()

    async def insert(self, job: RetrivalJob, request: RetrivalRequest) -> None:
        values = await job.to_polars()
        if not self.data.is_empty():
            self.data = self.data.vstack(values.select(self.data.columns))
        else:
            self.data = values

    async def upsert(self, job: RetrivalJob, request: RetrivalRequest) -> None:
        values = await job.to_lazy_polars()

        self.data = upsert_on_column(
            sorted(request.entity_names), new_data=values, existing_data=self.data.lazy()
        ).collect()

    async def overwrite(self, job: RetrivalJob, request: RetrivalRequest) -> None:
        self.data = await job.to_polars()

    async def write_polars(self, df: pl.LazyFrame) -> None:
        self.data = df.collect()

    def nearest_n_to(
        self, data: RetrivalJob, number_of_records: int, request: RetrivalRequest
    ) -> RetrivalJob:
        from aligned.retrival_job import RetrivalJob

        async def load() -> pl.LazyFrame:
            def first_embedding(features: set[Feature]) -> Feature | None:
                for feature in features:
                    if feature.dtype.is_embedding:
                        return feature
                return None

            embedding = first_embedding(data.request_result.features)
            assert embedding, 'Expected to a least find one embedding in the input data'

            df = await data.to_polars()

            def cosine_similarity(vector, candidate):
                import numpy as np

                vec1 = vector
                vec2 = np.array(candidate)

                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)

                return dot_product / (norm_vec1 * norm_vec2)

            result: pl.DataFrame | None = None

            org_columns = df.columns
            df_cols = len(df.columns)

            distance_key = 'distance'

            for item in df.iter_rows(named=True):
                most_similar = (
                    self.data.with_columns(
                        pl.col(embedding.name)
                        .map_elements(
                            lambda candidate: cosine_similarity(item[embedding.name], candidate),
                        )
                        .alias(distance_key)
                    )
                    .sort(distance_key, descending=True)
                    .head(number_of_records)
                    .select(pl.exclude(distance_key))
                )

                if df_cols > 1:
                    most_similar = most_similar.select(pl.exclude(org_columns)).hstack(
                        pl.DataFrame([item] * most_similar.height)
                        .select(org_columns)
                        .select(pl.exclude(embedding.name))
                    )

                if result is None:
                    result = most_similar
                else:
                    result = result.vstack(most_similar)

            if result is None:
                return pl.DataFrame().lazy()
            else:
                return result.lazy()

        return RetrivalJob.from_lazy_function(load, request)

    def with_view(self, view: 'CompiledFeatureView') -> 'InMemorySource':

        if self._vector_index_name is None:
            self._vector_index_name = view.name

        if self.data.is_empty():
            return InMemorySource.from_values({feat.name: [] for feat in view.entities.union(view.features)})
        return self

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls: type['InMemorySource'],
        facts: RetrivalJob,
        requests: list[tuple['InMemorySource', RetrivalRequest]],
    ) -> RetrivalJob:
        from aligned.local.job import FileFactualJob

        sources = {source.job_group_key() for source, _ in requests if isinstance(source, BatchDataSource)}
        if len(sources) != 1:
            raise NotImplementedError(
                f'Type: {cls} have not implemented how to load fact data with multiple sources.'
            )

        source, _ = requests[0]

        return FileFactualJob(source, [request for _, request in requests], facts)

    @staticmethod
    def from_values(values: dict[str, object]) -> 'InMemorySource':
        return InMemorySource(pl.DataFrame(values))

    @staticmethod
    def empty() -> 'InMemorySource':
        return InMemorySource.from_values({})
