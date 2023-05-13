from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from aligned import RedisConfig
from aligned.compiler.model import EntityDataSource, SqlEntityDataSource
from aligned.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aligned.enricher import Enricher
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FullExtractJob, RetrivalJob
from aligned.schemas.codable import Codable
from aligned.sources.psql import PostgreSQLConfig, PostgreSQLDataSource


@dataclass
class RedshiftSQLConfig(Codable):
    env_var: str
    schema: str | None = None

    @property
    def url(self) -> str:
        import os

        return os.environ[self.env_var]

    @property
    def psql_config(self) -> PostgreSQLConfig:

        return PostgreSQLConfig(self.env_var, self.schema)

    @staticmethod
    def from_url(url: str) -> RedshiftSQLConfig:
        import os

        os.environ['REDSHIFT_DATABASE'] = url.replace('redshift:', 'postgresql:')
        return RedshiftSQLConfig(env_var='REDSHIFT_DATABASE')

    def table(self, table: str, mapping_keys: dict[str, str] | None = None) -> RedshiftSQLDataSource:
        return RedshiftSQLDataSource(config=self, table=table, mapping_keys=mapping_keys or {})

    def data_enricher(
        self, name: str, sql: str, redis: RedisConfig, values: dict | None = None, lock_timeout: int = 60
    ) -> Enricher:
        from pathlib import Path

        from aligned.enricher import FileCacheEnricher, RedisLockEnricher, SqlDatabaseEnricher

        return FileCacheEnricher(
            timedelta(days=1),
            file=Path(f'./cache/{name}.parquet'),
            enricher=RedisLockEnricher(
                name, SqlDatabaseEnricher(self.url, sql, values), redis, timeout=lock_timeout
            ),
        )

    def entity_source(self, timestamp_column: str, sql: Callable[[str], str]) -> EntityDataSource:
        return SqlEntityDataSource(sql, self.url, timestamp_column)


@dataclass
class RedshiftSQLDataSource(BatchDataSource, ColumnFeatureMappable):

    config: RedshiftSQLConfig
    table: str
    mapping_keys: dict[str, str]

    type_name = 'redshift'

    def to_psql_source(self) -> PostgreSQLDataSource:
        return PostgreSQLDataSource(self.config.psql_config, self.table, self.mapping_keys)

    def job_group_key(self) -> str:
        return self.config.env_var

    def __hash__(self) -> int:
        return hash(self.table)

    def all_data(self, request: RetrivalRequest, limit: int | None) -> FullExtractJob:
        from aligned.psql.jobs import FullExtractPsqlJob

        return FullExtractPsqlJob(self, request, limit)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> DateRangeJob:
        from aligned.psql.jobs import DateRangePsqlJob, PostgreSQLDataSource

        source = PostgreSQLDataSource(self.config.psql_config, self.table, self.mapping_keys)

        return DateRangePsqlJob(source, start_date, end_date, request)

    @classmethod
    def multi_source_features_for(
        cls: type[RedshiftSQLDataSource],
        facts: RetrivalJob,
        requests: list[tuple[RedshiftSQLDataSource, RetrivalRequest]],
    ) -> RetrivalJob:
        from aligned.redshift.jobs import FactRedshiftJob

        return FactRedshiftJob(
            sources={request.location: source.to_psql_source() for source, request in requests},
            requests=[request for _, request in requests],
            facts=facts,
        )
