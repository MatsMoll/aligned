from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from aligned.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aligned.enricher import SqlDatabaseEnricher, StatisticEricher
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob, RetrivalJob
from aligned.schemas.codable import Codable

if TYPE_CHECKING:
    from aligned.compiler.feature_factory import FeatureFactory
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

        if 'PSQL_DATABASE' not in os.environ:
            os.environ['PSQL_DATABASE'] = url
        return PostgreSQLConfig(env_var='PSQL_DATABASE')

    @staticmethod
    def localhost(db: str, credentials: tuple[str, str] | None = None) -> PostgreSQLConfig:
        if credentials:
            return PostgreSQLConfig.from_url(f'postgresql://{credentials[0]}:{credentials[1]}@localhost/{db}')
        return PostgreSQLConfig.from_url(f'postgresql://localhost/{db}')

    def table(self, table: str, mapping_keys: dict[str, str] | None = None) -> PostgreSQLDataSource:
        return PostgreSQLDataSource(config=self, table=table, mapping_keys=mapping_keys or {})

    def data_enricher(self, sql: str, values: dict | None = None) -> Enricher:
        from aligned.enricher import SqlDatabaseEnricher

        return SqlDatabaseEnricher(self.env_var, sql, values)

    def entity_source(self, timestamp_column: str, sql: Callable[[str], str]) -> EntityDataSource:
        from aligned.compiler.model import SqlEntityDataSource

        return SqlEntityDataSource(sql, self.env_var, timestamp_column)

    def fetch(self, query: str) -> RetrivalJob:
        from aligned.psql.jobs import PostgreSqlJob

        return PostgreSqlJob(self, query)


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

        return SqlDatabaseEnricher(self.config.env_var, query)

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

        return SqlDatabaseEnricher(self.config.env_var, query)

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
    def multi_source_features_for(
        cls, facts: RetrivalJob, requests: list[tuple[PostgreSQLDataSource, RetrivalRequest]]
    ) -> FactualRetrivalJob:
        # Group based on config
        from aligned.psql.jobs import FactPsqlJob

        return FactPsqlJob(
            sources={request.location: source for source, request in requests},
            requests=[request for _, request in requests],
            facts=facts,
        )

    async def schema(self) -> dict[str, FeatureFactory]:
        import polars as pl

        import aligned.compiler.feature_factory as ff

        config = self.config
        schema = config.schema or 'public'
        table = self.table
        sql_query = f"""
SELECT column_name, data_type, character_maximum_length, is_nullable, column_default,
    CASE WHEN column_name IN (
        SELECT column_name
        FROM information_schema.key_column_usage
        WHERE constraint_name IN (
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_schema = '{schema}'
              AND table_name = '{table}'
              AND constraint_type = 'PRIMARY KEY'
        )
    ) THEN 'YES' ELSE 'NO' END AS is_primary_key
FROM information_schema.columns
WHERE table_schema = '{schema}'
  AND table_name = '{table}'"""
        df = pl.read_database(sql_query, connection_uri=self.config.url, engine='adbc')
        psql_types = {
            'uuid': ff.UUID(),
            'timestamp with time zone': ff.Timestamp(),
            'timestamp without time zone': ff.Timestamp(),
            'character varying': ff.String(),
            'text': ff.String(),
            'integer': ff.Int64(),
            'float': ff.Float(),
            'date': ff.Timestamp(),
            'boolean': ff.Bool(),
            'jsonb': ff.Json(),
            'smallint': ff.Int32(),
            'numeric': ff.Float(),
        }
        values = df.select(['column_name', 'data_type']).to_dicts()
        return {value['column_name']: psql_types[value['data_type']] for value in values}
