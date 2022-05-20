from dataclasses import dataclass
from typing import Callable
from datetime import timedelta
from redis.asyncio import Redis

from aladdin.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aladdin.codable import Codable
from aladdin.model import EntityDataSource, SqlEntityDataSource
from aladdin.enricher import FileCacheEnricher, RedisLockEnricher, SqlDatabaseEnricher, Enricher


@dataclass
class PostgreSQLConfig(Codable):
    env_var: str
    schema: str | None = None

    @property
    def url(self) -> str:
        import os
        return os.environ[self.env_var]

    @staticmethod
    def from_url(url: str) -> "PostgreSQLConfig":
        import os
        os.environ["PSQL_DATABASE"] = url
        return PostgreSQLConfig(env_var="PSQL_DATABASE")

    def table(self, table: str, mapping_keys: dict[str, str] | None = None) -> "PostgreSQLDataSource":
        return PostgreSQLDataSource(
            config=self,
            table=table,
            mapping_keys=mapping_keys or {}
        )

    def data_enricher(self, name: str, sql: str, redis_lock: Redis, values: dict | None = None) -> Enricher:
        from pathlib import Path
        return FileCacheEnricher(
            timedelta(days=1),
            file=Path(f"./cache/{name}.parquet"),
            enricher=RedisLockEnricher(
                name,
                SqlDatabaseEnricher(self.url, sql, values),
                redis_lock
            )
        )

    def entity_source(self, timestamp_column: str, sql: Callable[[str], str]) -> EntityDataSource:
        return SqlEntityDataSource(sql, self.url, timestamp_column)

@dataclass
class PostgreSQLDataSource(BatchDataSource, ColumnFeatureMappable):

    config: PostgreSQLConfig
    table: str
    mapping_keys: dict[str, str]

    type_name = "psql"

    def job_group_key(self) -> str:
        return self.config.env_var

    def __hash__(self) -> int:
        return hash(self.table)