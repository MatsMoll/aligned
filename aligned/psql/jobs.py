import logging
from abc import abstractproperty
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar

import pandas as pd

from aligned.psql.data_source import PostgreSQLConfig, PostgreSQLDataSource
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob, RetrivalJob
from aligned.schemas.feature import FeatureType

logger = logging.getLogger(__name__)

try:
    import dask.dataframe as dd
except ModuleNotFoundError:
    import pandas as dd

GenericDataFrame = TypeVar('GenericDataFrame', pd.DataFrame, dd.DataFrame)


@dataclass
class SQLQuery:
    sql: str
    values: dict[str, Any] | None = None


class PostgreSQLRetrivalJob(RetrivalJob):
    @abstractproperty
    def config(self) -> PostgreSQLConfig:
        pass

    def build_request(self) -> SQLQuery:
        raise NotImplementedError()

    async def _to_df(self) -> pd.DataFrame:
        sql_request = self.build_request()

        try:
            from sqlalchemy import create_engine
            from sqlalchemy.engine import Engine

            engine: Engine = create_engine(self.config.url).execution_options(autocommit=True)
            df = pd.read_sql_query(sql_request.sql, con=engine, params=sql_request.values)
            engine.dispose()
            return df
        except Exception as error:
            logger.info(sql_request.sql)
            logger.error(error)
            raise error

    async def _to_dask(self) -> dd.DataFrame:
        df = await self._to_df()
        return dd.from_pandas(df)


@dataclass
class FullExtractPsqlJob(PostgreSQLRetrivalJob, FullExtractJob):

    source: PostgreSQLDataSource
    request: RetrivalRequest
    limit: int | None = None

    @property
    def config(self) -> PostgreSQLConfig:
        return self.source.config

    def build_request(self) -> SQLQuery:

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

        return SQLQuery(
            sql=f'SELECT {column_select} FROM {schema}"{self.source.table}" {limit_query}',
        )


@dataclass
class DateRangePsqlJob(PostgreSQLRetrivalJob, DateRangeJob):

    source: PostgreSQLDataSource
    start_date: datetime
    end_date: datetime
    request: RetrivalRequest

    @property
    def config(self) -> PostgreSQLConfig:
        return self.source.config

    def build_request(self) -> SQLQuery:

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
        return SQLQuery(
            sql=(
                f'SELECT {column_select} FROM {schema}"{self.source.table}" WHERE'
                f' {event_timestamp_column} BETWEEN (:start_date) AND (:end_date)'
            ),
            values={'start_date': self.start_date, 'end_date': self.end_date},
        )


@dataclass
class FactPsqlJob(PostgreSQLRetrivalJob, FactualRetrivalJob):

    sources: dict[str, PostgreSQLDataSource]
    requests: list[RetrivalRequest]
    facts: dict[str, list]

    @property
    def config(self) -> PostgreSQLConfig:
        return list(self.sources.values())[0].config

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

    def build_request(self) -> SQLQuery:
        import numpy as np
        from jinja2 import BaseLoader, Environment

        template = Environment(loader=BaseLoader()).from_string(self.__sql_template())
        template_context: dict[str, Any] = {}

        final_select_names: set[str] = set()
        entity_types: dict[str, FeatureType] = {}
        has_event_timestamp = False

        for request in self.requests:
            final_select_names = final_select_names.union(
                {
                    f'{request.feature_view_name}_cte.{feature}'
                    for feature in request.all_required_feature_names
                }
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
        fact_df = pd.DataFrame(self.facts).replace(np.nan, None)

        number_of_values = max(len(values) for values in self.facts.values())
        # + 1 is needed as 0 is evaluated for null
        fact_df['row_id'] = list(range(1, number_of_values + 1))

        entity_type_list = [
            self.dtype_to_sql_type(entity_types.get(entity, FeatureType('').int32))
            for entity in fact_df.columns
        ]

        query_values = []
        all_entities = []
        for values in fact_df.values:
            row_placeholders = []
            for column_index, value in enumerate(values):
                row_placeholders.append(
                    {
                        'value': value,  # Could in theory lead to SQL injection (?)
                        'dtype': entity_type_list[column_index],
                    }
                )
                if fact_df.columns[column_index] not in all_entities:
                    all_entities.append(fact_df.columns[column_index])
            query_values.append(row_placeholders)

        feature_view_names: list[str] = list(self.sources.keys())
        # Add the joins to the fact

        tables = []
        for request in self.requests:
            source = self.sources[request.feature_view_name]
            field_selects = request.all_required_feature_names.union(
                {f'entities.{entity}' for entity in request.entity_names}
            ).union({'entities.row_id'})
            field_identifiers = source.feature_identifier_for(field_selects)
            selects = {
                feature if feature == db_field_name else f'{db_field_name} AS {feature}'
                for feature, db_field_name in zip(field_selects, field_identifiers)
            }

            entities = list(request.entity_names)
            entity_db_name = source.feature_identifier_for(entities)

            event_timestamp_clause = ''
            if request.event_timestamp:
                event_timestamp_column = source.feature_identifier_for([request.event_timestamp.name])[0]
                event_timestamp_clause = f'AND entities.event_timestamp >= ta.{event_timestamp_column}'

            join_conditions = [
                f'ta.{entity_db_name} = entities.{entity} {event_timestamp_clause}'
                for entity, entity_db_name in zip(entities, entity_db_name)
            ]
            tables.append(
                {
                    'name': source.table,
                    'joins': join_conditions,
                    'features': selects,
                    'fv': request.feature_view_name,
                }
            )

        template_context['selects'] = list(final_select_names)
        template_context['tables'] = tables
        template_context['joins'] = [
            f'INNER JOIN {feature_view}_cte ON {feature_view}_cte.row_id = entities.row_id'
            for feature_view in feature_view_names
        ]
        template_context['values'] = query_values
        template_context['entities'] = list(all_entities)

        # should insert the values as a value variable
        # As this can lead to sql injection
        return SQLQuery(sql=template.render(template_context))

    def __sql_template(self) -> str:
        return """
WITH entities (
    {{ entities | join(', ') }}
) AS (
VALUES {% for row in values %}
    ({% for value in row %}
        {% if value.value %}'{{value.value}}'::{{value.dtype}}{% else %}null::{{value.dtype}}{% endif %}
        {% if loop.last %}{% else %},{% endif %}{% endfor %}){% if loop.last %}{% else %},{% endif %}
{% endfor %}
),

{% for table in tables %}
    {{table.fv}}_cte AS (
        SELECT {{ table.features | join(', ') }}
        FROM entities
        LEFT JOIN {{table.name}} ta on {{ table.joins | join(' AND ') }}
    ){% if loop.last %}{% else %},{% endif %}
{% endfor %}

SELECT {{ selects | join(', ') }}
FROM entities
{{ joins | join('\n    ') }}

"""
