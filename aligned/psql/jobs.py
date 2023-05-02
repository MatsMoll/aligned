from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import polars as pl

from aligned.psql.data_source import PostgreSQLConfig, PostgreSQLDataSource
from aligned.request.retrival_request import RequestResult, RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob, RetrivalJob
from aligned.schemas.derivied_feature import AggregatedFeature, AggregateOver
from aligned.schemas.feature import FeatureLocation, FeatureType
from aligned.schemas.transformation import PsqlTransformation

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
        selection = self.selection
        # if not special operation e.g function. Then wrap in quotes
        if not ('(' in selection or '-' in selection or '.' in selection):
            selection = f'"{self.selection}"'

        if self.selection == self.alias:
            return f'{selection}'
        return f'{selection} AS "{self.alias}"'

    def __hash__(self) -> int:
        return hash(self.sql_select)


@dataclass
class SqlJoin:
    table: str
    conditions: list[str]


@dataclass
class TableFetch:
    name: str
    id_column: str
    table: str | TableFetch
    columns: set[SqlColumn]
    joins: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    order_by: str | None = field(default=None)

    def sql_query(self, distinct: str | None = None) -> str:
        # Select the core features
        wheres = ''
        order_by = ''
        group_by = ''
        select = 'SELECT'

        if distinct:
            select = f'SELECT DISTINCT ON ({distinct})'

        if self.conditions:
            wheres = 'WHERE ' + ' AND '.join(self.conditions)

        if self.order_by:
            order_by = 'ORDER BY ' + self.order_by

        if self.group_by:
            group_by = 'GROUP BY ' + ', '.join(self.group_by)

        table_columns = [col.sql_select for col in self.columns]

        if isinstance(self.table, TableFetch):
            from_sql = f'FROM ({self.table.sql_query()}) as entities'
        else:
            from_sql = f"""FROM entities
    LEFT JOIN "{ self.table }" ta ON { ' AND '.join(self.joins) }"""

        return f"""
    { select } { ', '.join(table_columns) }
    { from_sql }
    { wheres }
    { order_by }
    { group_by }"""


@dataclass
class PostgreSqlJob(RetrivalJob):

    config: PostgreSQLConfig
    query: str
    requests: list[RetrivalRequest] = field(default_factory=list)

    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.retrival_requests)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.requests

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.to_polars()
        return df.collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        try:
            return pl.read_sql(self.query, self.config.url).lazy()
        except Exception as e:
            logger.error(f'Error running query: {self.query}')
            logger.error(f'Error: {e}')
            raise e

    def describe(self) -> str:
        return f'PostgreSQL Job: \n{self.query}\n'


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

    def describe(self) -> str:
        return self.psql_job().describe()

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

        return f'SELECT {column_select} FROM {schema}"{self.source.table}" {limit_query}'


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

    def describe(self) -> str:
        if isinstance(self.facts, PostgreSqlJob):
            psql_job = self.build_sql_entity_query(self.facts)
            return f'Loading features for {self.facts.describe()}\n\nQuery: {psql_job}'
        else:
            raise ValueError(
                'Only PostgreSqlJob is supported as facts when describing,'
                f'but fetching features for facts: {self.facts.describe()}'
            )

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

        event_timestamp_clause: str | None = None
        if request.event_timestamp and entities_has_event_timestamp:
            event_timestamp_column = source.feature_identifier_for([request.event_timestamp.name])[0]
            event_timestamp_clause = f'entities.event_timestamp >= ta.{event_timestamp_column}'
            sort_query += f', {event_timestamp_column} DESC'

        join_conditions = [
            f'ta."{entity_db_name}" = entities.{entity}'
            for entity, entity_db_name in zip(entities, entity_db_name)
        ]
        if event_timestamp_clause:
            join_conditions.append(event_timestamp_clause)

        rename_fetch = TableFetch(
            name=f'{request.name}_cte',
            id_column='row_id',
            table=source.table,
            columns=selects,
            joins=join_conditions,
            order_by=sort_query,
        )

        derived_features = [
            feature
            for feature in request.derived_features
            if isinstance(feature.transformation, PsqlTransformation)
            and all([field in field_selects for field in feature.depending_on_names])
        ]
        if derived_features:
            derived_alias = source.feature_identifier_for([feature.name for feature in derived_features])
            derived_selects = {
                SqlColumn(feature.transformation.as_psql(), name)
                for feature, name in zip(derived_features, derived_alias)
                if isinstance(feature.transformation, PsqlTransformation)
            }.union({SqlColumn('*', '*')})

            return TableFetch(
                name=rename_fetch.name,
                id_column=rename_fetch.id_column,
                table=rename_fetch,
                columns=derived_selects,
            )
        else:
            return rename_fetch

    def sql_aggregated_request(
        self, window: AggregateOver, features: set[AggregatedFeature], request: RetrivalRequest
    ) -> TableFetch:
        source = self.sources[request.location]
        name = f'{request.name}_agg_cte'

        if not all(
            [isinstance(feature.derived_feature.transformation, PsqlTransformation) for feature in features]
        ):
            raise ValueError('All features must have a PsqlTransformation')

        group_by_names = {feature.name for feature in window.group_by}
        if window.window:
            time_window_config = window.window
            time_window = int(time_window_config.time_window.total_seconds())
            name = f'{request.name}_agg_{time_window}_cte'
            group_by_names = {'entities.row_id'}

        group_by_selects = {SqlColumn(feature, feature) for feature in group_by_names}

        aggregates = {
            SqlColumn(
                feature.derived_feature.transformation.as_psql(),
                feature.name,
            )
            for feature in features
        }

        id_column = window.group_by[0].name
        event_timestamp_clause: str | None = None
        if request.event_timestamp:
            id_column = 'row_id'
            group_by_names = {id_column}
            # Use row_id as the main join key
            event_timestamp_name = source.feature_identifier_for([request.event_timestamp.name])[0]
            if window.window:
                time_window_config = window.window
                window_in_seconds = int(time_window_config.time_window.total_seconds())
                event_timestamp_clause = (
                    f'ta.{event_timestamp_name} BETWEEN entities.event_timestamp'
                    f" - interval '{window_in_seconds} seconds' AND entities.event_timestamp"
                )
            else:
                event_timestamp_clause = f'ta.{event_timestamp_name} <= entities.event_timestamp'

        entities = list(request.entity_names)
        entity_db_name = source.feature_identifier_for(entities)
        join_conditions = [
            f'ta."{entity_db_name}" = entities.{entity}'
            for entity, entity_db_name in zip(entities, entity_db_name)
        ]
        if event_timestamp_clause:
            join_conditions.append(event_timestamp_clause)

        field_selects = request.all_required_feature_names.union({'entities.*'})
        field_identifiers = source.feature_identifier_for(field_selects)
        selects = {
            SqlColumn(db_field_name, feature)
            for feature, db_field_name in zip(field_selects, field_identifiers)
        }

        rename_table = TableFetch(
            name='ta',
            id_column='row_id',
            table=source.table,
            columns=selects,
            joins=join_conditions,
        )

        needed_derived_features = set()
        derived_map = request.derived_feature_map()

        for agg_feature in request.aggregated_features:
            for depended in agg_feature.depending_on_names:
                if depended in derived_map:
                    needed_derived_features.add(derived_map[depended])

        if needed_derived_features:
            features = {
                SqlColumn(feature.transformation.as_psql(), feature.name)
                for feature in needed_derived_features
            }.union(
                {SqlColumn('*', '*')}
            )  # Need the * in order to select the "original values"
            select_table = TableFetch(
                name='derived',
                id_column=rename_table.id_column,
                table=rename_table,
                columns=features,
            )
        else:
            select_table = rename_table

        return TableFetch(
            name=name,
            id_column=id_column,
            # Need to do a subquery, in order to renmae the core features
            table=select_table,
            columns=aggregates.union(group_by_selects),
            group_by=list(group_by_names),
        )

    def aggregated_values_from_request(self, request: RetrivalRequest) -> list[TableFetch]:

        aggregation_windows: dict[AggregateOver, set[AggregatedFeature]] = {}

        for aggregate in request.aggregated_features:
            if aggregate.aggregate_over not in aggregation_windows:
                aggregation_windows[aggregate.aggregate_over] = {aggregate}
            else:
                aggregation_windows[aggregate.aggregate_over].add(aggregate)

        fetches: list[TableFetch] = []
        supported_aggregation_features = set(request.feature_names).union(request.entity_names)
        for feature in request.derived_features:
            if isinstance(feature.transformation, PsqlTransformation):
                supported_aggregation_features.add(feature.name)

        for window, aggregates in aggregation_windows.items():
            needed_features = set()

            for agg in aggregates:
                for depended_feature in agg.derived_feature.depending_on:
                    needed_features.add(depended_feature.name)

            missing_features = needed_features - supported_aggregation_features
            if not missing_features:
                fetches.append(self.sql_aggregated_request(window, aggregates, request))
            else:
                raise ValueError(
                    f'Only SQL aggregates are supported at the moment. Missing features {missing_features}'
                )
                # fetches.append(self.fetch_all_aggregate_values(window, aggregates, request))

        return fetches

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
        aggregates: list[TableFetch] = []
        for request in self.requests:
            fetch = self.value_selection(request, has_event_timestamp)
            tables.append(fetch)
            aggregate_fetches = self.aggregated_values_from_request(request)
            aggregates.extend(aggregate_fetches)
            for aggregate in aggregate_fetches:
                final_select_names = final_select_names.union(
                    {column.alias for column in aggregate.columns if column.alias != aggregate.id_column}
                )

        joins = '\n    '.join(
            [
                f'LEFT JOIN {feature_view}_cte ON {feature_view}_cte.row_id = entities.row_id'
                for feature_view in feature_view_names
            ]
        )
        if aggregates:
            joins += '\n    '
            joins += '\n    '.join(
                [
                    f'LEFT JOIN {table.name} ON {table.name}.{table.id_column} = entities.{table.id_column}'
                    for table in aggregates
                ]
            )

        entity_values = self.build_entities_from_values(query_values)

        return self.generate_query(
            entity_columns=list(all_entities),
            entity_query=entity_values,
            tables=tables,
            aggregates=aggregates,
            final_select=list(final_select_names),
            final_joins=joins,
        )

    def build_entities_from_values(self, values: list[list[SqlValue]]) -> str:
        query = 'VALUES '
        for row in values:
            query += '\n    ('
            for value in row:
                query += value.to_sql + ', '
            query = query[:-2]
            query += '),'
        query = query[:-1]
        return query

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
        aggregates: list[TableFetch] = []
        for request in self.requests:
            all_entities.update(request.entity_names)

            if request.aggregated_features:
                aggregate_fetches = self.aggregated_values_from_request(request)
                aggregates.extend(aggregate_fetches)
                for aggregate in aggregate_fetches:
                    final_select_names = final_select_names.union(
                        {column.alias for column in aggregate.columns if column.alias != 'entites.row_id'}
                    )
            else:
                fetch = self.value_selection(request, has_event_timestamp)
                tables.append(fetch)
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
            [f'LEFT JOIN {table.name} ON {table.name}.row_id = entities.row_id' for table in tables]
        )
        if aggregates:
            joins += '\n    '
            joins += '\n    '.join(
                [
                    f'LEFT JOIN {table.name} ON {table.name}.{table.id_column} = entities.{table.id_column}'
                    for table in aggregates
                ]
            )

        return self.generate_query(
            entity_columns=all_entities_list,
            entity_query=entity_query,
            tables=tables,
            aggregates=aggregates,
            final_select=list(final_select_names),
            final_joins=joins,
        )

    def generate_query(
        self,
        entity_columns: list[str],
        entity_query: str,
        tables: list[TableFetch],
        aggregates: list[TableFetch],
        final_select: list[str],
        final_joins: str,
    ) -> str:

        query = f"""
WITH entities (
    { ', '.join(entity_columns) }
) AS (
    { entity_query }
),"""

        # Select the core features
        for table in tables:
            query += f"""
{table.name} AS (
    { table.sql_query(distinct='entities.row_id') }
    ),"""

        # Select the aggregate features
        for table in aggregates:
            query += f"""
{table.name} AS (
    { table.sql_query() }
    ),"""

        query = query[:-1]  # Dropping the last comma
        query += f"""
SELECT { ', '.join(final_select) }
FROM entities
{ final_joins }
"""
        return query
