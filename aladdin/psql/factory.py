from datetime import datetime

from aladdin.job_factory import JobFactory
from aladdin.psql.data_source import PostgreSQLDataSource
from aladdin.psql.jobs import DateRangePsqlJob, FactPsqlJob, FullExtractPsqlJob
from aladdin.request.retrival_request import RetrivalRequest


class PostgresJobFactory(JobFactory):

    source = PostgreSQLDataSource

    def all_data(
        self, source: PostgreSQLDataSource, request: RetrivalRequest, limit: int | None
    ) -> FullExtractPsqlJob:
        return FullExtractPsqlJob(source, request, limit)

    def all_between_dates(
        self,
        source: PostgreSQLDataSource,
        request: RetrivalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> DateRangePsqlJob:
        return DateRangePsqlJob(source, start_date, end_date, request)

    def _facts(
        self,
        facts: dict[str, list],
        requests: dict[PostgreSQLDataSource, RetrivalRequest],
    ) -> FactPsqlJob:
        # Group based on config
        return FactPsqlJob(
            sources={request.feature_view_name: source for source, request in requests.items()},
            requests=list(requests.values()),
            facts=facts,
        )
