from datetime import datetime

from aladdin.job_factory import JobFactory
from aladdin.psql.jobs import DateRangePsqlJob, FactPsqlJob, FullExtractPsqlJob
from aladdin.redshift.data_source import RedshiftSQLDataSource
from aladdin.request.retrival_request import RetrivalRequest


class RedshiftJobFactory(JobFactory):

    source = RedshiftSQLDataSource

    def all_data(
        self, source: RedshiftSQLDataSource, request: RetrivalRequest, limit: int | None
    ) -> FullExtractPsqlJob:
        return FullExtractPsqlJob(source, request, limit)

    def all_between_dates(
        self,
        source: RedshiftSQLDataSource,
        request: RetrivalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> DateRangePsqlJob:
        raise NotImplementedError()

    def _facts(
        self,
        facts: dict[str, list],
        requests: dict[RedshiftSQLDataSource, RetrivalRequest],
    ) -> FactPsqlJob:
        # Group based on config
        return FactPsqlJob(
            facts=facts,
            sources={request.feature_view_name: source for source, request in requests.items()},
            requests=list(requests.values()),
        )
