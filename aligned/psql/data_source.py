from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from aligned.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aligned.enricher import SqlDatabaseEnricher, StatisticEricher
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob
from aligned.schemas.codable import Codable

if TYPE_CHECKING:
    from aligned.enricher import Enricher, TimespanSelector
    from aligned.entity_data_source import EntityDataSource


@dataclass
class PostgreSQLConfig(Codable):
    env_var: str
    schema: str | None = None

    @property
    def url(self) -> str:
        import os

        return os.environ[self.env_var]

    @staticmethod
    def from_url(url: str) -> PostgreSQLConfig:
        import os

        os.environ['PSQL_DATABASE'] = url
        return PostgreSQLConfig(env_var='PSQL_DATABASE')

    @staticmethod
    def localhost(db: str) -> PostgreSQLConfig:
        return PostgreSQLConfig.from_url(f'postgresql://localhost/{db}')

    def table(self, table: str, mapping_keys: dict[str, str] | None = None) -> PostgreSQLDataSource:
        return PostgreSQLDataSource(config=self, table=table, mapping_keys=mapping_keys or {})

    def data_enricher(self, sql: str, values: dict | None = None) -> Enricher:
        from aligned.enricher import SqlDatabaseEnricher

        return SqlDatabaseEnricher(self.env_var, sql, values)

    def entity_source(self, timestamp_column: str, sql: Callable[[str], str]) -> EntityDataSource:
        from aligned.model import SqlEntityDataSource

        return SqlEntityDataSource(sql, self.env_var, timestamp_column)


@dataclass
class PostgreSQLDataSource(BatchDataSource, ColumnFeatureMappable, StatisticEricher):

    config: PostgreSQLConfig
    table: str
    mapping_keys: dict[str, str]

    type_name = 'psql'

    def job_group_key(self) -> str:
        return self.config.env_var

    def __hash__(self) -> int:
        return hash(self.table)

    def mean(
        self, columns: set[str], time: TimespanSelector | None = None, limit: int | None = None
    ) -> Enricher:
        reverse_map = {value: key for key, value in self.mapping_keys.items()}
        sql_columns = ', '.join([f'AVG({reverse_map.get(column, column)}) AS {column}' for column in columns])

        query = f'SELECT {sql_columns} FROM {self.table}'
        if time:
            seconds = time.timespand.total_seconds()
            query += f' WHERE {time.time_column} >= NOW() - interval \'{seconds} seconds\''
        if limit and isinstance(limit, int):
            query += f' LIMIT {limit}'

        return SqlDatabaseEnricher(self.config.url, query)

    def std(
        self, columns: set[str], time: TimespanSelector | None = None, limit: int | None = None
    ) -> Enricher:
        reverse_map = {value: key for key, value in self.mapping_keys.items()}
        sql_columns = ', '.join(
            [f'STDDEV({reverse_map.get(column, column)}) AS {column}' for column in columns]
        )

        query = f'SELECT {sql_columns} FROM {self.table}'
        if time:
            seconds = time.timespand.total_seconds()
            query += f' WHERE {time.time_column} >= NOW() - interval \'{seconds} seconds\''
        if limit and isinstance(limit, int):
            query += f' LIMIT {limit}'

        return SqlDatabaseEnricher(self.config.url, query)

    def all_data(self, request: RetrivalRequest, limit: int | None) -> FullExtractJob:
        from aligned.psql.jobs import FullExtractPsqlJob

        return FullExtractPsqlJob(self, request, limit)

    def all_between_dates(
        self,
        request: RetrivalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> DateRangeJob:
        from aligned.psql.jobs import DateRangePsqlJob

        return DateRangePsqlJob(self, start_date, end_date, request)

    @classmethod
    def feature_for(
        cls, facts: dict[str, list], requests: dict[PostgreSQLDataSource, RetrivalRequest]
    ) -> FactualRetrivalJob:
        # Group based on config
        from aligned.psql.jobs import FactPsqlJob

        return FactPsqlJob(
            sources={request.feature_view_name: source for source, request in requests.items()},
            requests=list(requests.values()),
            facts=facts,
        )
