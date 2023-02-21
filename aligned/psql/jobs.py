import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import polars as pl

from aligned.psql.data_source import PostgreSQLConfig, PostgreSQLDataSource
from aligned.request.retrival_request import RequestResult, RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob, RetrivalJob
from aligned.schemas.derivied_feature import AggregationConfig
from aligned.schemas.feature import FeatureLocation, FeatureType

logger = logging.getLogger(__name__)


@dataclass
class SQLQuery:
    sql: str


@dataclass
class SqlColumn:
    selection: str
    alias: str

    @property
    def sql_select(self) -> str:
        if self.selection == self.alias:
            return f'{self.selection}'
        return f'{self.selection} AS {self.alias}'

    def __hash__(self) -> int:
        return hash(self.sql_select)


@dataclass
class SqlJoin:
    table: str
    conditions: list[str]


@dataclass
class TableFetch:
    name: str
    table_name: str
    columns: set[SqlColumn]
    joins: list[SqlJoin] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    order_by: str = field(default='')


@dataclass
class PostgreSqlJob(RetrivalJob):

    config: PostgreSQLConfig
    query: str
    retrival_requests: list[RetrivalRequest] = field(default_factory=list)

    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.retrival_requests)

    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.retrival_requests

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.to_polars()
        return df.collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        return pl.read_sql(self.query, self.config.url).lazy()


@dataclass
class FullExtractPsqlJob(FullExtractJob):

    source: PostgreSQLDataSource
    request: RetrivalRequest
    limit: int | None = None

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request(self.request)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return [self.request]

    @property
    def config(self) -> PostgreSQLConfig:
        return self.source.config

    async def to_pandas(self) -> pd.DataFrame:
        return await self.psql_job().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        return await self.psql_job().to_polars()

    def psql_job(self) -> PostgreSqlJob:
        return PostgreSqlJob(self.config, self.build_request())

    def build_request(self) -> str:

        all_features = [
            feature.name for feature in list(self.request.all_required_features.union(self.request.entities))
        ]
        sql_columns = self.source.feature_identifier_for(all_features)
        columns = [
            f'"{sql_col}" AS {alias}' if sql_col != alias else sql_col
            for sql_col, alias in zip(sql_columns, all_features)
        ]
        column_select = ', '.join(columns)
        schema = f'{self.config.schema}.' if self.config.schema else ''

        limit_query = ''
        if self.limit:
            limit_query = f'LIMIT {int(self.limit)}'

        f'SELECT {column_select} FROM {schema}"{self.source.table}" {limit_query}',


@dataclass
class DateRangePsqlJob(DateRangeJob):

    source: PostgreSQLDataSource
    start_date: datetime
    end_date: datetime
    request: RetrivalRequest

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request(self.request)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return [self.request]

    @property
    def config(self) -> PostgreSQLConfig:
        return self.source.config

    async def to_pandas(self) -> pd.DataFrame:
        return await self.psql_job().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        return await self.psql_job().to_polars()

    def psql_job(self) -> PostgreSqlJob:
        return PostgreSqlJob(self.config, self.build_request())

    def build_request(self) -> str:

        if not self.request.event_timestamp:
            raise ValueError('Event timestamp is needed in order to run a data range job')

        event_timestamp_column = self.source.feature_identifier_for([self.request.event_timestamp.name])[0]
        all_features = [
            feature.name for feature in list(self.request.all_required_features.union(self.request.entities))
        ]
        sql_columns = self.source.feature_identifier_for(all_features)
        columns = [
            f'"{sql_col}" AS {alias}' if sql_col != alias else sql_col
            for sql_col, alias in zip(sql_columns, all_features)
        ]
        column_select = ', '.join(columns)
        schema = f'{self.config.schema}.' if self.config.schema else ''
        start_date = self.start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_date = self.end_date.strftime('%Y-%m-%d %H:%M:%S')

        return (
            f'SELECT {column_select} FROM {schema}"{self.source.table}" WHERE'
            f' {event_timestamp_column} BETWEEN \'{start_date}\' AND \'{end_date}\''
        )


@dataclass
class SqlValue:
    value: str | None
    data_type: str

    @property
    def to_sql(self) -> str:
        if self.value:
            return f"'{self.value}'::{self.data_type}"
        else:
            return f'NULL::{self.data_type}'


@dataclass
class FactPsqlJob(FactualRetrivalJob):
    """Fetches features for defined facts within a postgres DB

    It is supported to fetch from different tables, in one request
    This is hy the `source` property is a dict with sources

    NB: It is expected that the data sources are for the same psql instance
    """

    sources: dict[FeatureLocation, PostgreSQLDataSource]
    requests: list[RetrivalRequest]
    facts: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.requests

    @property
    def config(self) -> PostgreSQLConfig:
        return list(self.sources.values())[0].config

    async def to_pandas(self) -> pd.DataFrame:
        job = await self.psql_job()
        return await job.to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        job = await self.psql_job()
        return await job.to_polars()

    async def psql_job(self) -> PostgreSqlJob:
        if isinstance(self.facts, PostgreSqlJob):
            return PostgreSqlJob(self.config, self.build_sql_entity_query(self.facts))
        return PostgreSqlJob(self.config, await self.build_request())

    def dtype_to_sql_type(self, dtype: object) -> str:
        if isinstance(dtype, str):
            return dtype
        if dtype == FeatureType('').string:
            return 'text'
        if dtype == FeatureType('').uuid:
            return 'uuid'
        if dtype == FeatureType('').int32 or dtype == FeatureType('').int64:
            return 'integer'
        if dtype == FeatureType('').datetime:
            return 'TIMESTAMP WITH TIME ZONE'
        return 'uuid'

    def value_selection(self, request: RetrivalRequest, entities_has_event_timestamp: bool) -> TableFetch:

        source = self.sources[request.location]

        entity_selects = {f'entities.{entity}' for entity in request.entity_names}
        field_selects = request.all_required_feature_names.union(entity_selects).union({'entities.row_id'})
        field_identifiers = source.feature_identifier_for(field_selects)
        selects = {
            SqlColumn(db_field_name, feature)
            for feature, db_field_name in zip(field_selects, field_identifiers)
        }

        entities = list(request.entity_names)
        entity_db_name = source.feature_identifier_for(entities)
        sort_query = 'entities.row_id'

        event_timestamp_clause = ''
        if request.event_timestamp and entities_has_event_timestamp:
            event_timestamp_column = source.feature_identifier_for([request.event_timestamp.name])[0]
            event_timestamp_clause = f'AND entities.event_timestamp >= ta.{event_timestamp_column}'
            sort_query += f', {event_timestamp_column} DESC'

        join_conditions = [
            f'ta.{entity_db_name} = entities.{entity} {event_timestamp_clause}'
            for entity, entity_db_name in zip(entities, entity_db_name)
        ]

        return TableFetch(
            name=f'{request.name}_cte',
            table_name=source.table,
            columns=selects,
            joins=join_conditions,
            order_by=sort_query,
        )

    def aggregated_values_from_request(self, request: RetrivalRequest, table: str) -> list[str]:

        tables: dict[AggregationConfig, dict] = {}

        for aggregate in request.aggregated_features:
            aggregate.aggregate_over.time_window

        return {
            'name': 'featurestore',
            'features': [
                {
                    'aggregation': 'sum',
                    'name': 'feature1',
                }
            ],
            'joins': ['table1'],
            'filter': 'table1.id = table2.id',
            'group_by': ['table1.id'],
        }

    async def build_request(self) -> str:
        import numpy as np

        final_select_names: set[str] = set()
        entity_types: dict[str, FeatureType] = {}
        has_event_timestamp = False

        for request in self.requests:
            final_select_names = final_select_names.union(
                {f'{request.location.name}_cte.{feature}' for feature in request.all_required_feature_names}
            )
            final_select_names = final_select_names.union(
                {f'entities.{entity}' for entity in request.entity_names}
            )
            for entity in request.entities:
                entity_types[entity.name] = entity.dtype
            if request.event_timestamp:
                has_event_timestamp = True

        if has_event_timestamp:
            final_select_names.add('event_timestamp')
            entity_types['event_timestamp'] = FeatureType('').datetime

        # Need to replace nan as it will not be encoded
        fact_df = await self.facts.to_pandas()
        fact_df = fact_df.replace(np.nan, None)

        number_of_values = fact_df.shape[0]
        # + 1 is needed as 0 is evaluated for null
        fact_df['row_id'] = list(range(1, number_of_values + 1))

        entity_type_list = [
            self.dtype_to_sql_type(entity_types.get(entity, FeatureType('').int32))
            for entity in fact_df.columns
        ]

        query_values: list[list[SqlValue]] = []
        all_entities = []
        for values in fact_df.values:
            row_placeholders = []
            for column_index, value in enumerate(values):
                row_placeholders.append(SqlValue(value, entity_type_list[column_index]))
                if fact_df.columns[column_index] not in all_entities:
                    all_entities.append(fact_df.columns[column_index])
            query_values.append(row_placeholders)

        feature_view_names: list[str] = [location.name for location in self.sources.keys()]
        # Add the joins to the fact

        tables: list[TableFetch] = []
        all_entities = set()
        for request in self.requests:
            fetch = self.value_selection(request, has_event_timestamp)
            tables.append(fetch)
            all_entities.update(request.entity_names)

        joins = [
            f'INNER JOIN {feature_view}_cte ON {feature_view}_cte.row_id = entities.row_id'
            for feature_view in feature_view_names
        ]
        entity_values = self.build_entities_from_values(query_values)

        return self.generate_query(
            entity_columns=list(all_entities),
            entity_query=entity_values,
            tables=tables,
            final_select=list(final_select_names),
            final_joins=joins,
        )

    def build_entities_from_values(self, values: list[list[SqlValue]]) -> str:
        query = '('
        for row in values:
            query += '\n    ('
            for value in row:
                query += value.to_sql() + ', '
            query = query[:-1]
            query += '),'
        query = query[:-1]
        return query + ')'

    def build_sql_entity_query(self, sql_facts: PostgreSqlJob) -> str:

        final_select_names: set[str] = set()
        has_event_timestamp = False
        all_entities = set()

        if 'event_timestamp' in sql_facts.query:
            has_event_timestamp = True
            all_entities.add('event_timestamp')

        for request in self.requests:
            final_select_names = final_select_names.union(
                {f'entities.{entity}' for entity in request.entity_names}
            )

        if has_event_timestamp:
            final_select_names.add('event_timestamp')

        # Add the joins to the fact

        tables: list[TableFetch] = []
        for request in self.requests:
            fetch = self.value_selection(request, has_event_timestamp)
            tables.append(fetch)
            all_entities.update(request.entity_names)
            final_select_names = final_select_names.union(
                {f'{fetch.name}.{feature}' for feature in request.all_required_feature_names}
            )

        all_entities_list = list(all_entities)
        all_entities_str = ', '.join(all_entities_list)
        all_entities_list.append('row_id')
        entity_query = (
            f'SELECT {all_entities_str}, ROW_NUMBER() OVER (ORDER BY '
            f'{list(request.entity_names)[0]}) AS row_id FROM ({sql_facts.query}) AS entities'
        )
        joins = '\n    '.join(
            [f'INNER JOIN {table.name} ON {table.name}.row_id = entities.row_id' for table in tables]
        )

        return self.generate_query(
            entity_columns=all_entities_list,
            entity_query=entity_query,
            tables=tables,
            final_select=list(final_select_names),
            final_joins=joins,
        )

    def generate_query(
        self,
        entity_columns: list[str],
        entity_query: str,
        tables: list[TableFetch],
        final_select: list[str],
        final_joins: str,
    ) -> str:

        aggregations: list[TableFetch] = []

        query = f"""
WITH entities (
    { ', '.join(entity_columns) }
) AS (
    { entity_query }
),"""

        # Select the core features
        for table in tables:
            wheres = ''
            if table.conditions:
                wheres = 'WHERE ' + ' AND '.join(table.conditions)

            table_columns = [col.sql_select for col in table.columns]
            query += f"""
{table.name} AS (
        SELECT DISTINCT ON (entities.row_id) { ', '.join(table_columns) }
        FROM entities
        LEFT JOIN { table.table_name } ta ON { ' AND '.join(table.joins) }
        { wheres }
        ORDER BY { table.order_by }
    ),"""

        # Add aggregation values
        for agg in aggregations:
            wheres = ''
            if agg.conditions:
                wheres = 'WHERE ' + ' AND '.join(agg.conditions)

            query += f"""
{agg.name} AS (
        SELECT { ', '.join(table_columns) }
        FROM entities
        LEFT JOIN { agg.table_name } ta ON { ' AND '.join(agg.joins) }
        { wheres }
        GROUP BY { ', '.join(agg.group_by) }
    ),"""

        query = query[:-1]  # Dropping the last comma
        query += f"""
SELECT { ', '.join(final_select) }
FROM entities
{ final_joins }
"""
        return query

    def __sql_entities_template(self) -> str:
        return """
WITH entities (
    {{ entities | join(', ') }}
) AS (
    {{ entities_sql }}
),
{% for agg in aggregates %}
    {{agg.name}}_agg AS (
        SELECT entities.row_id, {{agg.aggregate}} as {{agg.feature}}
        FROM entities
        LEFT JOIN {{agg.table}} ta on {{ agg.joins | join(' AND ') }}
        GROUP BY 1
    )
{% endfor %}
solutions_agg AS (
    SELECT entities."taskID", array_to_string(array_agg(ts.solution), '\n') as all_solutions FROM entities
    LEFT JOIN "TaskSolution" ts ON ts."taskID" = entities."taskID"
    GROUP BY 1
)
{% for table in tables %}
    {{table.fv}}_cte AS (
        SELECT DISTINCT ON (entities.row_id) {{ table.features | join(', ') }}
        FROM entities
        LEFT JOIN {{table.name}} ta on {{ table.joins | join(' AND ') }}
        ORDER BY {{table.sort_query}}
    ){% if loop.last %}{% else %},{% endif %}
{% endfor %}

SELECT {{ selects | join(', ') }}
FROM entities
{{ joins | join('\n    ') }}

"""
