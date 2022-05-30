from datetime import datetime
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.job_factory import JobFactory
from aladdin.psql.data_source import PostgreSQLConfig
from aladdin.psql.jobs import DateRangePsqlJob, FullExtractPsqlJob, FactPsqlJob
from aladdin.redshift.data_source import RedshiftSQLDataSource

class RedshiftJobFactory(JobFactory):

    source = RedshiftSQLDataSource

    def all_data(self, source: RedshiftSQLDataSource, request: RetrivalRequest, limit: int | None) -> FullExtractPsqlJob:
        return FullExtractPsqlJob(source, request, limit)

    def all_between_dates(self, source: RedshiftSQLDataSource, request: RetrivalRequest, start_date: datetime, end_date: datetime) -> DateRangePsqlJob:
        raise NotImplementedError()

    def _facts(self, facts: dict[str, list], requests: dict[RedshiftSQLDataSource, RetrivalRequest]) -> FactPsqlJob:
        for data_source in requests.keys():
            if not isinstance(data_source, RedshiftSQLDataSource):
                raise ValueError(f"Only {self.source} is supported, recived: {data_source}")
            config = data_source.config

        # Group based on config
        return FactPsqlJob(
            # Reusing the config as a psql config
            # As redshift is a fork from psql
            config=PostgreSQLConfig(
                config.env_var,
                schema=config.schema
            ),
            facts=facts,
            sources={request.feature_view_name: source for source, request in requests.items()},
            requests=list(requests.values())
        )