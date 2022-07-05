from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from aladdin.codable import Codable
from aladdin.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aladdin.enricher import SqlDatabaseEnricher, StatisticEricher

if TYPE_CHECKING:
    from aladdin.enricher import Enricher
    from aladdin.entity_data_source import EntityDataSource


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
        from aladdin.enricher import SqlDatabaseEnricher

        return SqlDatabaseEnricher(self.env_var, sql, values)

    def entity_source(self, timestamp_column: str, sql: Callable[[str], str]) -> EntityDataSource:
        from aladdin.model import SqlEntityDataSource

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

    def mean(self, columns: set[str]) -> Enricher:
        sql_columns = ', '.join([f'AVG({column}) AS {column}' for column in columns])
        return SqlDatabaseEnricher(self.config.url, f'SELECT {sql_columns} FROM {self.table}')

    def std(self, columns: set[str]) -> Enricher:
        sql_columns = ', '.join([f'STDDEV({column}) AS {column}' for column in columns])
        return SqlDatabaseEnricher(self.config.url, f'SELECT {sql_columns} FROM {self.table}')
