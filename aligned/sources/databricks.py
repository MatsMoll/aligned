from datetime import datetime
from dataclasses import dataclass
from aligned.feature_source import WritableFeatureSource


try:
    from pyspark import SparkConf
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import col

    import pandas as pd
    import polars as pl
    from aligned.data_source.batch_data_source import BatchDataSource
    from aligned.retrival_job import RetrivalJob, RetrivalRequest, RequestResult

    @dataclass
    class SparkConfig:
        spark_conf: dict[str, str] | None = None
        region: str | None = None

    def session_from_config(
        config: dict,
    ) -> SparkSession:
        session = SparkSession.getActiveSession()
        if session:
            return session

        return SparkSession.builder.config(
            conf=SparkConf().setAll([(k, v) for k, v in config.items()])
        ).getOrCreate()

    class DatabaseTable:
        catalog: str
        schema: str
        table: str

        @property
        def identifier(self) -> str:
            return f'{self.catalog}.{self.schema}.{self.table}'

    @dataclass
    class AllRequest(RetrivalJob):

        table: DatabaseTable
        request: RetrivalRequest

        def request_result(self) -> RequestResult:
            return self.request.request_result

        def retrival_requests(self) -> list[RetrivalRequest]:
            return [self.request]

        def job_group_key(self) -> str:
            return 'all_unity_catalog_table'

        def describe(self) -> str:
            return f"Loading all data from {self.table.identifier} with columns {self.request.core_column_names}."

        async def to_pandas(self) -> pd.DataFrame:
            raise ValueError('You need to use `to_spark` as a spark session is required')

        async def to_spark(self, spark: SparkSession) -> DataFrame:
            table = self.table.identifier
            df = spark.read.table(table)
            return df.select(self.request.core_column_names)

        async def to_lazy_polars(self) -> pl.LazyFrame:
            raise ValueError('You need to use `to_spark` as a spark session is required')

    @dataclass
    class BetweenDatesRequest(RetrivalJob):

        table: DatabaseTable
        request: RetrivalRequest

        from_date: datetime
        to_date: datetime

        def request_result(self) -> RequestResult:
            return self.request.request_result

        def retrival_requests(self) -> list[RetrivalRequest]:
            return [self.request]

        def job_group_key(self) -> str:
            return 'all_unity_catalog_table'

        def describe(self) -> str:
            return f"Loading all data from {self.table.identifier} with columns {self.request.core_column_names}."

        async def to_pandas(self) -> pd.DataFrame:
            raise ValueError('You need to use `to_spark` as a spark session is required')

        async def to_spark(self, spark: SparkSession) -> DataFrame:
            table = self.table.identifier
            event_timestamp = self.request.event_timestamp

            if not event_timestamp:
                raise ValueError('Event timestamp is required for between dates request')

            df = spark.read.table(table).select(self.request.core_column_names)
            df = df.filter(col(event_timestamp.name).between(self.from_date, self.to_date))
            return df

        async def to_lazy_polars(self) -> pl.LazyFrame:
            raise ValueError('You need to use `to_spark` as a spark session is required')

    class SparkTable(BatchDataSource, WritableFeatureSource):

        table: DatabaseTable
        type_name: str = 'spark_table'

        def job_group_key(self) -> str:
            return 'unity_catalog_table'

        def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:
            job = AllRequest(self.table, request)
            if limit:
                job = job.limit(limit)
            return job

        def all_between_dates(
            self, request: RetrivalRequest, start_date: datetime, end_date: datetime
        ) -> RetrivalJob:
            # Should ideally be a AllRequest with a filter
            # But the filter API needs to be improved first
            return BetweenDatesRequest(self.table, request, start_date, end_date)

        @classmethod
        def multi_source_features_for(
            cls: type['SparkTable'], facts: RetrivalJob, requests: list[tuple['SparkTable', RetrivalRequest]]
        ) -> RetrivalJob:
            raise NotImplementedError(cls)

        async def overwrite(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:
            if len(requests) > 1:
                raise ValueError('Only one request is allowed')

            request = requests[0]
            subset = job.select(request.features_to_include)

            try:
                session = session_from_config({})
                df = await subset.to_spark(session)
            except NotImplementedError:
                spark = session_from_config({})
                df = spark.createDataFrame(await subset.to_pandas())

            df.write.mode('overwrite').saveAsTable(self.table.identifier)

except ModuleNotFoundError:
    pass
