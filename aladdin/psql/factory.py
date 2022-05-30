from datetime import datetime
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.job_factory import JobFactory
from aladdin.psql.data_source import PostgreSQLDataSource
from aladdin.psql.jobs import DateRangePsqlJob, FullExtractPsqlJob, FactPsqlJob

class PostgresJobFactory(JobFactory):

    source = PostgreSQLDataSource

    def all_data(self, source: PostgreSQLDataSource, request: RetrivalRequest, limit: int | None) -> FullExtractPsqlJob:
        return FullExtractPsqlJob(source, request, limit)

    def all_between_dates(self, source: PostgreSQLDataSource, request: RetrivalRequest, start_date: datetime, end_date: datetime) -> DateRangePsqlJob:
        raise NotImplementedError()

    def _facts(self, facts: dict[str, list], requests: dict[PostgreSQLDataSource, RetrivalRequest]) -> FactPsqlJob:
        for data_source in requests.keys():
            if not isinstance(data_source, PostgreSQLDataSource):
                raise ValueError(f"Only {self.source} is supported, recived: {data_source}")
            config = data_source.config

        # Group based on config
        return FactPsqlJob(
            config=config,
            facts=facts,
            sources={request.feature_view_name: source for source, request in requests.items()},
            requests=list(requests.values())
        )