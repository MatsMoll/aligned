from dataclasses import dataclass
from pandas import DataFrame
from typing import Any
from datetime import datetime
from aladdin.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aladdin.feature import FeatureType
from aladdin.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob, RetrivalJob
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.job_factory import JobFactory
from aladdin.codable import Codable

@dataclass
class PostgreSQLConfig(Codable):
    url: str
    schema: str | None

    def table(self, table_name: str, mapping_keys: dict[str, str] | None = None) -> "PostgreSQLDataSource":
        return PostgreSQLDataSource(
            config=self,
            table=table_name,
            mapping_keys=mapping_keys or {}
        )

    def data_enricher(self, sql: str):
        raise NotImplementedError()

@dataclass
class SQLQuery:
    sql: str
    values: dict[str, Any] | None = None

@dataclass
class PostgreSQLDataSource(BatchDataSource, ColumnFeatureMappable):

    config: PostgreSQLConfig
    table: str
    mapping_keys: dict[str, str]

    type_name = "psql"

    def job_group_key(self) -> str:
        return self.config.url

    def __hash__(self) -> int:
        return hash(self.table)


@dataclass
class PostgreSQLRetrivalJob(RetrivalJob):

    config: PostgreSQLConfig

    async def fetch_data(self) -> DataFrame:
        sql_request = self.build_request()
        from databases import Database
        try:
            async with Database(self.config.url) as db:
                records = await db.fetch_all(query=sql_request.sql, values=sql_request.values)
        except Exception as e:
            print(sql_request.sql)
            print(e)
        return DataFrame.from_records([dict(record) for record in records])


    def build_request(self) -> SQLQuery:
        raise NotImplementedError()



class FullExtractPsqlJob(PostgreSQLRetrivalJob, FullExtractJob):
    
    source: PostgreSQLDataSource

    def build_request(self) -> SQLQuery:
        
        columns = self.source.columns_for(list(self.request.all_required_features))
        column_select = ", ".join(columns)

        return SQLQuery(
            sql=f"SELECT {column_select} FROM {self.source.table}",
        )

@dataclass
class DateRangePsqlJob(PostgreSQLRetrivalJob, DateRangeJob):
    
    source: PostgreSQLDataSource
    start_date: datetime
    end_date: datetime
    request: RetrivalRequest

    def build_request(self) -> SQLQuery:

        columns = self.source.columns_for(list(self.request.all_required_features))
        column_select = ", ".join(columns)

        return SQLQuery(
            sql=f"SELECT {column_select} FROM {self.source.table} WHERE {self.source.event_timestamp_column} BETWEEN ((:start_date), (:end_date))", 
            values={
                "start_date": self.start_date,
                "end_date": self.end_date
        })

@dataclass
class FactPsqlJob(PostgreSQLRetrivalJob, FactualRetrivalJob):

    grouped: dict[PostgreSQLDataSource, RetrivalRequest]
    facts: dict[str, list]

    def dtype_to_sql_type(self, dtype: type) -> str:
        if isinstance(dtype, str):
            return dtype
        if dtype == FeatureType("").string:
            return "text"
        if dtype == FeatureType("").uuid:
            return "uuid"
        if dtype == FeatureType("").int32 or dtype == FeatureType("").int64:
            return "integer"
        return "uuid"

    def build_request(self) -> SQLQuery:
        import pandas as pd
        final_select_names = set()
        entity_types: dict[str, str] = {}
        for request in self.grouped.values():
            final_select_names = final_select_names.union(request.all_required_feature_names)
            final_select_names = final_select_names.union({f"entities.{entity}" for entity in request.entity_names})
            for entity in request.entities:
                entity_types[entity.name] = entity.dtype
        final_select = ", ".join(final_select_names)

        fact_df = DataFrame(self.facts)
        number_of_values = max([len(values) for values in self.facts.values()])
        fact_df["row_id"] = list(range(number_of_values))

        entity_names = ", ".join(list(fact_df.columns))
        entity_types = [self.dtype_to_sql_type(entity_types.get(entity, FeatureType("").int32)) for entity in fact_df.columns]

        encoded_entity_values = "),\n    (".join([
            ", ".join([
                f"'{value}'::{entity_types[index]}" if value != None and not pd.isna(value) else f"null::{entity_types[index]}" for index, value in enumerate(values)
            ]) 
            for values in fact_df.values
        ])
        encoded_entity_values = f"({encoded_entity_values})"
        feature_view_names: list[str] = [source.table for source in self.grouped.keys()]
        join_clauses = "\n".join([f"INNER JOIN {feature_view}_cte ON {feature_view}_cte.row_id = entities.row_id" for feature_view in feature_view_names])
        # Add the joins to the fact

        sub_queries = ",\n".join([self.sub_query(source, request) for source, request in self.grouped.items()])

        fact_query = f"""
WITH entities ({entity_names}) AS (
VALUES {encoded_entity_values}
),

{sub_queries}

SELECT {final_select} FROM entities {join_clauses}
"""
        # should insert the values as a value variable
        # As this can lead to sql injection
        return SQLQuery(
            sql=fact_query
        )

    def sub_query(self, source: PostgreSQLDataSource, request: RetrivalRequest) -> str:        
        field_selects = request.all_required_feature_names + [f"entities.{entity}" for entity in request.entity_names] + ["entities.row_id"]
        field_identifiers = source.feature_identifier_for(field_selects)
        selects = [feature if feature == db_field_name else f"{db_field_name} AS {feature}" for feature, db_field_name in zip(field_selects, field_identifiers)]

        entities = list(request.entity_names)
        entity_db_name = source.feature_identifier_for(entities)

        join_conditions = [f"ta.{entity_db_name} = entities.{entity}" for entity, entity_db_name in zip(entities, entity_db_name)]
        join_clause = " AND ".join(join_conditions)
        select_clause = ", ".join(selects)
        table_name = source.table
        return f"""
{table_name}_cte AS (
    SELECT {select_clause}
    FROM entities
    LEFT JOIN {table_name} ta on {join_clause}
)
"""
    async def _to_df(self) -> DataFrame:
        return await self.fetch_data()

    async def to_arrow(self) -> DataFrame:
        return await super().to_arrow()


class PostgresJobFactory(JobFactory):

    source = PostgreSQLDataSource

    def all_data(self, source: PostgreSQLDataSource) -> FullExtractPsqlJob:
        return super().all_data()

    def all_between_dates(self, source: PostgreSQLDataSource, request: RetrivalRequest, start_date: datetime, end_date: datetime) -> DateRangePsqlJob:
        return DateRangePsqlJob(
            config=self.config,
            request=request,
            source=self,
            start_date=start_date,
            end_date=end_date
        )

    def _facts(self, facts: dict[str, list], requests: dict[PostgreSQLDataSource, RetrivalRequest]) -> FactPsqlJob:
        for data_source in requests.keys():
            if not isinstance(data_source, PostgreSQLDataSource):
                raise ValueError("Only PostgreSQLDataSource is supported")
            config = data_source.config

        # Group based on config

        return FactPsqlJob(
            config=config,
            facts=facts,
            grouped=requests            
        )