from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    ColumnFeatureMappable,
)
from aligned.request.retrieval_request import RetrievalRequest
from aligned.retrieval_job import RetrievalJob
from aligned.schemas.codable import Codable
from aligned.sources.psql import PostgreSQLConfig, PostgreSQLDataSource

if TYPE_CHECKING:
    from aligned.schemas.feature import Feature


@dataclass
class RedshiftListReference(Codable):
    """
    A class representing a one to many relationship.
    This can simulate how a list datatype
    """

    table_schema: str
    table_name: str
    value_column: str
    id_column: str
    join_column: str | None = None


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

        if "REDSHIFT_DATABASE" not in os.environ:
            os.environ["REDSHIFT_DATABASE"] = url.replace("redshift:", "postgresql:")
        return RedshiftSQLConfig(env_var="REDSHIFT_DATABASE")

    def table(
        self,
        table: str,
        mapping_keys: dict[str, str] | None = None,
        list_references: dict[str, RedshiftListReference] | None = None,
    ) -> RedshiftSQLDataSource:
        return RedshiftSQLDataSource(
            config=self,
            table=table,
            mapping_keys=mapping_keys or {},
            list_references=list_references or {},
        )

    def with_schema(self, name: str) -> RedshiftSQLConfig:
        return RedshiftSQLConfig(env_var=self.env_var, schema=name)

    def fetch(self, query: str) -> RetrievalJob:
        from aligned.redshift.jobs import PostgreSqlJob

        return PostgreSqlJob(self.psql_config, query)


@dataclass
class RedshiftSQLDataSource(CodableBatchDataSource, ColumnFeatureMappable):
    config: RedshiftSQLConfig
    table: str
    mapping_keys: dict[str, str]
    list_references: dict[str, RedshiftListReference] = field(default_factory=dict)

    type_name = "redshift"

    def to_psql_source(self) -> PostgreSQLDataSource:
        return PostgreSQLDataSource(
            self.config.psql_config, self.table, self.mapping_keys
        )

    def job_group_key(self) -> str:
        return self.config.env_var

    def contains_config(self, config: Any) -> bool:
        return (
            isinstance(config, RedshiftSQLConfig)
            and config.env_var == self.config.env_var
        )

    def __hash__(self) -> int:
        return hash(self.table)

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        from aligned.psql.jobs import build_full_select_query_psql
        from aligned.redshift.sql_job import RedshiftSqlJob

        source = PostgreSQLDataSource(
            self.config.psql_config, self.table, self.mapping_keys
        )
        return RedshiftSqlJob(
            config=self.config,
            query=build_full_select_query_psql(source, request, limit),
            requests=[request],
        )

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        from aligned.redshift.sql_job import RedshiftSqlJob
        from aligned.psql.jobs import build_date_range_query_psql

        source = PostgreSQLDataSource(
            self.config.psql_config, self.table, self.mapping_keys
        )
        return RedshiftSqlJob(
            config=self.config,
            query=build_date_range_query_psql(source, request, start_date, end_date),
            requests=[request],
        )

    @classmethod
    def multi_source_features_for(
        cls: type[RedshiftSQLDataSource],
        facts: RetrievalJob,
        requests: list[tuple[RedshiftSQLDataSource, RetrievalRequest]],
    ) -> RetrievalJob:
        from aligned.redshift.jobs import FactRedshiftJob

        return FactRedshiftJob(
            sources={request.location: source for source, request in requests},
            requests=[request for _, request in requests],
            facts=facts,
        )

    async def freshness(self, feature: Feature) -> datetime | None:
        f"SELECT MAX({feature.name})"
        return await super().freshness(feature)
