import uuid

import polars as pl
from aligned.data_file import DataFileReference, upsert_on_column
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.feature_source import WritableFeatureSource
from aligned.retrival_job import RetrivalJob, RetrivalRequest


class InMemorySource(BatchDataSource, DataFileReference, WritableFeatureSource):
    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data
        self.job_key = str(uuid.uuid4())

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
