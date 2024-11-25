from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from logging import getLogger

import polars as pl

from aligned.request.retrival_request import RequestResult, RetrivalRequest
from aligned.retrival_job import RetrivalJob
from aligned.sources.redshift import RedshiftSQLConfig

if TYPE_CHECKING:
    import pandas as pd

logger = getLogger(__name__)


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
        return f'{self.selection} AS "{self.alias}"'

    def __hash__(self) -> int:
        return hash(self.sql_select)


@dataclass
class SqlJoin:
    table: str
    conditions: list[str]


@dataclass
class TableFetch:
    """
    A configuration of an SQL query on a table
    """

    name: str
    id_column: str
    table: str | TableFetch
    columns: set[SqlColumn]
    schema: str | None = field(default=None)
    joins: list[str] = field(default_factory=list)
    join_tables: list[tuple[TableFetch, str]] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    order_by: str | None = field(default=None)

    def sql_query(self, distinct: str | None = None) -> str:
        return redshift_table_fetch(self, distinct)


def select_table(table: TableFetch) -> str:
    if isinstance(table.table, TableFetch):
        raise ValueError('Do not support TableFetch in this select')
    wheres = ''
    order_by = ''
    group_by = ''
    from_table = 'FROM '

    columns = [col.sql_select for col in table.columns]
    select = f'SELECT {', '.join(columns)}'

    if table.conditions:
        wheres = 'WHERE ' + ' AND '.join(table.conditions)

    if table.order_by:
        order_by = 'ORDER BY ' + table.order_by

    if table.group_by:
        group_by = 'GROUP BY ' + ', '.join(table.group_by)

    if table.schema:
        from_table += f'{table.schema}.'

    from_table += f'"{table.table}"'

    return f"""
    {select}
    {from_table}
    {wheres}
    {order_by}
    {group_by}
    """


def redshift_table_fetch(fetch: TableFetch, distinct: str | None = None) -> str:
    wheres = ''
    order_by = ''
    group_by = ''
    select = 'SELECT'

    if fetch.conditions:
        wheres = 'WHERE ' + ' AND '.join(fetch.conditions)

    if fetch.order_by:
        order_by = 'ORDER BY ' + fetch.order_by

    if fetch.group_by:
        group_by = 'GROUP BY ' + ', '.join(fetch.group_by)

    table_columns = [col.sql_select for col in fetch.columns]

    if isinstance(fetch.table, TableFetch):
        sub_query = redshift_table_fetch(fetch.table)
        from_sql = f'FROM ({sub_query}) as entities'
    else:
        schema = f'{fetch.schema}.' if fetch.schema else ''
        from_sql = f"""FROM entities
    LEFT JOIN {schema}"{ fetch.table }" ta ON { ' AND '.join(fetch.joins) }"""
        if fetch.join_tables:
            for join_table, join_condition in fetch.join_tables:
                from_sql += f"""
                LEFT JOIN (
                    {select_table(join_table)}
                ) AS {join_table.name} ON {join_condition}
                """

    if distinct:
        aliases = [col.alias for col in fetch.columns]
        return f"""
        SELECT { ', '.join(aliases) }
        FROM (
    { select } { ', '.join(table_columns) },
        ROW_NUMBER() OVER(
            PARTITION BY entities.row_id
            { order_by }
        ) AS row_number
        { from_sql }
        { wheres }
        { order_by }
        { group_by }
    ) AS entities
    WHERE row_number = 1"""
    else:
        return f"""
    { select } { ', '.join(table_columns) }
    { from_sql }
    { wheres }
    { order_by }
    { group_by }"""


@dataclass
class RedshiftSqlJob(RetrivalJob):

    config: RedshiftSQLConfig
    query: str
    requests: list[RetrivalRequest]

    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.retrival_requests)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.requests

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.to_lazy_polars()
        return df.collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        try:
            return pl.read_sql(self.query, self.config.url).lazy()
        except Exception as e:
            logger.error(f'Error running query: {self.query}')
            logger.error(f'Error: {e}')
            raise e

    def describe(self) -> str:
        return f'RedshiftSql Job: \n{self.query}\n'
