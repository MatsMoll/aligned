from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    ColumnFeatureMappable,
)
from aligned.feature_source import WritableFeatureSource
from aligned.request.retrieval_request import RetrievalRequest
from aligned.retrieval_job import RetrievalJob
from aligned.schemas.codable import Codable
from datetime import datetime

from aligned.schemas.feature import FeatureType

if TYPE_CHECKING:
    from aligned.schemas.feature import Feature


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

        if "PSQL_DATABASE" not in os.environ:
            os.environ["PSQL_DATABASE"] = url
        return PostgreSQLConfig(env_var="PSQL_DATABASE")

    @staticmethod
    def localhost(
        db: str, credentials: tuple[str, str] | None = None
    ) -> PostgreSQLConfig:
        if credentials:
            return PostgreSQLConfig.from_url(
                f"postgresql://{credentials[0]}:{credentials[1]}@localhost/{db}"
            )
        return PostgreSQLConfig.from_url(f"postgresql://localhost/{db}")

    def table(
        self, table: str, mapping_keys: dict[str, str] | None = None
    ) -> PostgreSQLDataSource:
        return PostgreSQLDataSource(
            config=self, table=table, mapping_keys=mapping_keys or {}
        )

    def fetch(self, query: str) -> RetrievalJob:
        from aligned.psql.jobs import PostgreSqlJob

        return PostgreSqlJob(self, query)


@dataclass
class PostgreSQLDataSource(
    CodableBatchDataSource, ColumnFeatureMappable, WritableFeatureSource
):
    config: PostgreSQLConfig
    table: str
    mapping_keys: dict[str, str]

    type_name = "psql"

    def source_id(self) -> str:
        return f"{self.config.env_var}/{self.table}"

    def job_group_key(self) -> str:
        return self.config.env_var

    def contains_config(self, config: Any) -> bool:
        return (
            isinstance(config, PostgreSQLConfig)
            and config.env_var == self.config.env_var
        )

    def __hash__(self) -> int:
        return hash(self.table)

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        from aligned.psql.jobs import build_full_select_query_psql, PostgreSqlJob

        return PostgreSqlJob(
            config=self.config,
            query=build_full_select_query_psql(self, request, limit),
            requests=[request],
        )

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        from aligned.psql.jobs import build_date_range_query_psql, PostgreSqlJob

        return PostgreSqlJob(
            config=self.config,
            query=build_date_range_query_psql(self, request, start_date, end_date),
            requests=[request],
        )

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls,
        facts: RetrievalJob,
        requests: list[tuple[PostgreSQLDataSource, RetrievalRequest]],
    ) -> RetrievalJob:
        # Group based on config
        from aligned.psql.jobs import FactPsqlJob

        return FactPsqlJob(
            sources={request.location: source for source, request in requests},
            requests=[request for _, request in requests],
            facts=facts,
        )

    async def schema(self) -> dict[str, FeatureType]:
        import polars as pl

        config = self.config
        schema = config.schema or "public"
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
        df = pl.read_database_uri(sql_query, uri=self.config.url, engine="adbc")
        psql_types = {
            "uuid": FeatureType.uuid(),
            "timestamp with time zone": FeatureType.datetime(),
            "timestamp without time zone": FeatureType.datetime(None),
            "character varying": FeatureType.string(),
            "text": FeatureType.string(),
            "integer": FeatureType.int32(),
            "float": FeatureType.floating_point(),
            "date": FeatureType.date(),
            "boolean": FeatureType.boolean(),
            "jsonb": FeatureType.json(),
            "smallint": FeatureType.int16(),
            "numeric": FeatureType.floating_point(),
        }
        values = df.select(["column_name", "data_type"]).to_dicts()
        return {
            value["column_name"]: psql_types[value["data_type"]] for value in values
        }

    async def freshness(self, feature: Feature) -> datetime | None:
        import polars as pl

        value = pl.read_database_uri(
            query=f"SELECT MAX({feature.name}) as freshness FROM {self.table}",
            uri=self.config.url,
        )["freshness"].max()

        if value:
            if isinstance(value, datetime):
                return value
            else:
                raise ValueError(f"Unsupported freshness value {value}")
        else:
            return None

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        data = await job.to_lazy_polars()
        data.select(request.all_returned_columns).collect().write_database(
            self.table, connection=self.config.url, if_table_exists="append"
        )

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        import asyncpg

        all_columns = request.all_returned_columns
        entities = list(request.entity_names)

        all_col_sql = ", ".join(all_columns)
        update_col_statement = ", ".join(
            [f"{col} = EXCLUDED.{col}" for col in all_columns if col not in entities]
        )
        col_parameter_str = ", ".join([f"${i + 1}" for i in range(len(all_columns))])
        entity_sql = ", ".join(entities)

        df = await job.to_pandas()

        if self.config.schema:
            table = f"{self.config.schema}.{self.table}"
        else:
            table = self.table

        query = f"""INSERT INTO {table}({all_col_sql})
VALUES ({col_parameter_str})
ON CONFLICT ({entity_sql})
DO UPDATE SET {update_col_statement}
"""

        conn = await asyncpg.connect(self.config.url)
        await conn.executemany(query, df[all_columns].to_numpy())
