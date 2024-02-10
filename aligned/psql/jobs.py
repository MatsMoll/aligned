from __future__ import annotations

from typing import TYPE_CHECKING
import logging
from dataclasses import dataclass, field

import pandas as pd
import polars as pl

from aligned.request.retrival_request import RequestResult, RetrivalRequest
from aligned.retrival_job import RetrivalJob
from aligned.schemas.derivied_feature import AggregatedFeature, AggregateOver, DerivedFeature
from aligned.schemas.feature import FeatureLocation, FeatureType, Feature
from aligned.schemas.transformation import PsqlTransformation
from aligned.sources.psql import PostgreSQLConfig, PostgreSQLDataSource

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class SQLQuery:
    sql: str


@dataclass
class SqlColumn:
    selection: str
    alias: str

    @property
    def sql_select(self) -> str:
        return psql_select_column(self)

    def __hash__(self) -> int:
        return hash(self.sql_select)


def psql_select_column(column: SqlColumn) -> str:
    selection = column.selection
    # if not special operation e.g function. Then wrap in quotes
    if not ('(' in selection or '-' in selection or '.' in selection or ' ' in selection or selection == '*'):
        selection = f'"{column.selection}"'

    if column.selection == column.alias:
        return f'{selection}'
    return f'{selection} AS "{column.alias}"'


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
        return psql_table_fetch(self, distinct)


def psql_table_fetch(fetch: TableFetch, distinct: str | None = None) -> str:
    # Select the core features
    wheres = ''
    order_by = ''
    group_by = ''
    select = 'SELECT'

    if distinct:
        select = f'SELECT DISTINCT ON ({distinct})'

    if fetch.conditions:
        wheres = 'WHERE ' + ' AND '.join(fetch.conditions)

    if fetch.order_by:
        order_by = 'ORDER BY ' + fetch.order_by

    if fetch.group_by:
        group_by = 'GROUP BY ' + ', '.join(fetch.group_by)

    table_columns = [col.sql_select for col in fetch.columns]

    if isinstance(fetch.table, TableFetch):
        from_sql = f'FROM ({fetch.table.sql_query()}) as entities'
    else:
        from_sql = f"""FROM entities
LEFT JOIN "{ fetch.table }" ta ON { ' AND '.join(fetch.joins) }"""

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

    def will_load_list_feature(self) -> bool:
        for request in self.requests:
            for feature in request.all_features:
                if feature.dtype == FeatureType.array():
                    return True
        return False

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.to_lazy_polars()
        return df.collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        try:
            return pl.read_database(self.query, self.config.url).lazy()
        except Exception as e:
            logger.error(f'Error running query: {self.query}')
            logger.error(f'Error: {e}')
            raise e

    def describe(self) -> str:
        return f'PostgreSQL Job: \n{self.query}\n'

    def filter(self, condition: str | Feature | DerivedFeature) -> RetrivalJob:

        query = f'SELECT * FROM ({self.query}) as values WHERE '

        if isinstance(condition, str):
            query += condition
        elif isinstance(condition, DerivedFeature) and isinstance(
            condition.transformation, PsqlTransformation
        ):
            query += condition.transformation.as_psql()
        elif isinstance(condition, Feature):
            query += condition.name
        else:
            raise ValueError(f'Unable to filter on psql job with {condition}')

        return PostgreSqlJob(self.config, query, self.requests)


def build_full_select_query_psql(
    source: PostgreSQLDataSource, request: RetrivalRequest, limit: int | None
) -> str:
    """
    Generates the SQL query needed to select all features related to a psql data source
    """
    all_features = [feature.name for feature in list(request.all_required_features.union(request.entities))]
    sql_columns = source.feature_identifier_for(all_features)
    columns = [
        f'"{sql_col}" AS {alias}' if sql_col != alias else sql_col
        for sql_col, alias in zip(sql_columns, all_features)
    ]
    column_select = ', '.join(columns)

    config = source.config
    schema = f'{config.schema}.' if config.schema else ''

    limit_query = ''
    if limit:
        limit_query = f'LIMIT {int(limit)}'

    return f'SELECT {column_select} FROM {schema}"{source.table}" {limit_query}'


def build_date_range_query_psql(
    source: PostgreSQLDataSource, request: RetrivalRequest, start_date: datetime, end_date: datetime
) -> str:
    if not request.event_timestamp:
        raise ValueError('Event timestamp is needed in order to run a data range job')

    event_timestamp_column = source.feature_identifier_for([request.event_timestamp.name])[0]
    all_features = [feature.name for feature in list(request.all_required_features.union(request.entities))]
    sql_columns = source.feature_identifier_for(all_features)
    columns = [
        f'"{sql_col}" AS {alias}' if sql_col != alias else sql_col
        for sql_col, alias in zip(sql_columns, all_features)
    ]
    column_select = ', '.join(columns)

    config = source.config
    schema = f'{config.schema}.' if config.schema else ''
    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

    return (
        f'SELECT {column_select} FROM {schema}"{source.table}" WHERE'
        f' {event_timestamp_column} BETWEEN \'{start_date_str}\' AND \'{end_date_str}\''
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
class FactPsqlJob(RetrivalJob):
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
        from aligned.retrival_job import LiteralDictJob

        if isinstance(self.facts, PostgreSqlJob):
            psql_job = self.build_sql_entity_query(self.facts)
            return f'Loading features for {self.facts.describe()}\n\nQuery: {psql_job}'
        elif isinstance(self.facts, LiteralDictJob):
            psql_job = self.build_request_from_facts(pl.DataFrame(self.facts.data).lazy())
            return f'Loading features from dicts \n\nQuery: {psql_job}'
        else:
            return f'Loading features from {self.facts.describe()}, and its related features'

    async def to_pandas(self) -> pd.DataFrame:
        job = await self.psql_job()
        return await job.to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        job = await self.psql_job()
        return await job.to_lazy_polars()

    async def psql_job(self) -> PostgreSqlJob:
        if isinstance(self.facts, PostgreSqlJob):
            return PostgreSqlJob(self.config, self.build_sql_entity_query(self.facts), self.retrival_requests)
        entities = await self.build_request()
        return PostgreSqlJob(self.config, entities, self.retrival_requests)

    def ignore_event_timestamp(self) -> RetrivalJob:
        return FactPsqlJob(
            self.sources, [request.without_event_timestamp() for request in self.requests], self.facts
        )

    def dtype_to_sql_type(self, dtype: object) -> str:
        if isinstance(dtype, str):
            return dtype
        if dtype == FeatureType.string():
            return 'text'
        if dtype == FeatureType.uuid():
            return 'uuid'
        if dtype == FeatureType.int32() or dtype == FeatureType.int64():
            return 'integer'
        if dtype == FeatureType.datetime():
            return 'TIMESTAMP WITH TIME ZONE'
        return 'uuid'

    def value_selection(self, request: RetrivalRequest, entities_has_event_timestamp: bool) -> TableFetch:

        source = self.sources[request.location]

        entity_selects = {f'entities.{entity}' for entity in request.entity_names}
        field_selects = list(
            request.all_required_feature_names.union(entity_selects).union({'entities.row_id'})
        )
        field_identifiers = source.feature_identifier_for(field_selects)
        selects = {
            SqlColumn(db_field_name, feature)
            for feature, db_field_name in zip(field_selects, field_identifiers)
        }

        entities = list(request.entity_names)
        entity_db_name = source.feature_identifier_for(entities)
        sort_query = 'entities.row_id'

        event_timestamp_clause: str | None = None
        if request.event_timestamp_request:
            timestamp = request.event_timestamp_request.event_timestamp
            event_timestamp_column = source.feature_identifier_for([timestamp.name])[0]
            sort_query += f', ta.{event_timestamp_column} DESC'

            if request.event_timestamp_request.entity_column:
                entity_column = request.event_timestamp_request.entity_column
                event_timestamp_clause = f'entities.{entity_column} >= ta.{event_timestamp_column}'

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

        id_column = 'row_id'
        # id_column = window.group_by[0].name
        event_timestamp_clause: str | None = None
        if request.event_timestamp_request and request.event_timestamp_request.entity_column:
            timestamp = request.event_timestamp_request.event_timestamp
            entity_column = request.event_timestamp_request.entity_column
            group_by_names = {id_column}
            # Use row_id as the main join key
            event_timestamp_name = source.feature_identifier_for([timestamp.name])[0]
            if window.window:
                time_window_config = window.window
                window_in_seconds = int(time_window_config.time_window.total_seconds())
                event_timestamp_clause = (
                    f'ta.{event_timestamp_name} BETWEEN entities.{entity_column}'
                    f" - interval '{window_in_seconds} seconds' AND entities.{entity_column}"
                )
            else:
                event_timestamp_clause = f'ta.{event_timestamp_name} <= entities.{entity_column}'

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

        aggregation_windows = request.aggregate_over()

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
        facts = await self.facts.to_lazy_polars()
        return self.build_request_from_facts(facts)

    def build_request_from_facts(self, facts: pl.LazyFrame) -> str:

        final_select_names: set[str] = set()
        all_entities = {'row_id'}
        entity_types: dict[str, FeatureType] = {'row_id': FeatureType.int64()}
        has_event_timestamp = False

        for request in self.requests:
            final_select_names = final_select_names.union(
                {f'entities.{entity}' for entity in request.entity_names}
            )
            for entity in request.entities:
                entity_types[entity.name] = entity.dtype
                all_entities.add(entity.name)

            if request.event_timestamp_request and request.event_timestamp_request.entity_column:
                entity_column = request.event_timestamp_request.entity_column
                has_event_timestamp = True
                entity_types[entity_column] = FeatureType.datetime()
                all_entities.add(entity_column)
                final_select_names.add(f'entities.{entity_column}')

        all_entities_list = list(all_entities)

        # Need to replace nan as it will not be encoded
        fact_df = facts.with_row_count(name='row_id', offset=1).collect()

        entity_type_list = {
            entity: self.dtype_to_sql_type(entity_types.get(entity, FeatureType.int32()))
            for entity in all_entities
        }

        query_values: list[list[SqlValue]] = []
        for values in fact_df[all_entities_list].to_dicts():
            row_placeholders = []
            for key, value in values.items():
                row_placeholders.append(SqlValue(value, entity_type_list[key]))

            query_values.append(row_placeholders)
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
                        {column.alias for column in aggregate.columns if column.alias != 'entities.row_id'}
                    )
            else:
                fetch = self.value_selection(request, has_event_timestamp)
                tables.append(fetch)
                final_select_names = final_select_names.union(
                    {f'{fetch.name}.{feature}' for feature in request.all_required_feature_names}
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

        entity_values = self.build_entities_from_values(query_values)

        return self.generate_query(
            entity_columns=all_entities_list,
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

        for request in self.requests:
            final_select_names = final_select_names.union(
                {f'entities.{entity}' for entity in request.entity_names}
            )
            if request.event_timestamp_request and request.event_timestamp_request.entity_column:
                entity_column = request.event_timestamp_request.entity_column

                if entity_column in sql_facts.query:
                    has_event_timestamp = True
                    all_entities.add(entity_column)
                    final_select_names.add(f'entities.{entity_column}')

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
                        {column.alias for column in aggregate.columns if column.alias != 'entities.row_id'}
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
