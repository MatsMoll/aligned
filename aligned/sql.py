from __future__ import annotations
from typing import TYPE_CHECKING
from collections import defaultdict
import polars as pl
from sqlglot import parse_one, exp

if TYPE_CHECKING:
    from aligned.feature_store import ContractStore
    from aligned.request.retrieval_request import RetrievalRequest


def sql_to_polars(sql: str) -> pl.Expr:
    parsed = parse_one(sql)

    if isinstance(parsed, exp.Where):
        return glot_to_polars(parsed.this)

    return glot_to_polars(parsed)


def glot_to_polars(glot_exp: exp.Expression) -> pl.Expr:
    if isinstance(glot_exp, exp.Literal):
        val = glot_exp.this
        if glot_exp.args.get("is_string"):
            return pl.lit(val)
        else:
            return pl.lit(int(val))

    if isinstance(glot_exp, exp.Is):
        left = glot_to_polars(glot_exp.left)
        if isinstance(glot_exp.right, exp.Null):
            return left.is_null()
        if isinstance(glot_exp.right, exp.Not) and isinstance(
            glot_exp.right.this, exp.Null
        ):
            return left.is_not_null()

    if isinstance(glot_exp, exp.Null):
        return pl.lit(None)

    if isinstance(glot_exp, exp.Alias):
        return glot_to_polars(glot_exp.this).alias(glot_exp.alias)
    if isinstance(glot_exp, exp.Column):
        return pl.col(glot_exp.name)

    if isinstance(glot_exp, exp.Add):
        return glot_to_polars(glot_exp.left) + glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, exp.Sub):
        return glot_to_polars(glot_exp.left) - glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, exp.Mul):
        return glot_to_polars(glot_exp.left) * glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, exp.Div):
        return glot_to_polars(glot_exp.left) / glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, exp.LT):
        return glot_to_polars(glot_exp.left) < glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, exp.LTE):
        return glot_to_polars(glot_exp.left) <= glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, exp.GT):
        return glot_to_polars(glot_exp.left) > glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, exp.GTE):
        return glot_to_polars(glot_exp.left) >= glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, (exp.EQ, exp.Is)):
        return glot_to_polars(glot_exp.left) == glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, exp.NEQ):
        return glot_to_polars(glot_exp.left) != glot_to_polars(glot_exp.right)
    if isinstance(glot_exp, exp.Mod):
        return glot_to_polars(glot_exp.left).mod(glot_to_polars(glot_exp.right))
    if isinstance(glot_exp, exp.And):
        return glot_to_polars(glot_exp.left).and_(glot_to_polars(glot_exp.right))
    if isinstance(glot_exp, exp.Or):
        return glot_to_polars(glot_exp.left).or_(glot_to_polars(glot_exp.right))

    if isinstance(glot_exp, exp.Not):
        return glot_to_polars(glot_exp.this).not_()
    if isinstance(glot_exp, exp.Neg):
        return glot_to_polars(glot_exp.this).not_()

    if isinstance(glot_exp, exp.In):
        return glot_to_polars(glot_exp.this).is_in(
            [glot_to_polars(val) for val in glot_exp.expressions]
        )
    raise NotImplementedError(f"Unable to find pl expr for {glot_exp}")


def extract_raw_values(query: str) -> pl.DataFrame | None:
    glot_exp = parse_one(query)
    vals = list(glot_exp.find_all(exp.Values))

    if len(vals) != 1:
        return None

    val = vals[0]

    raw_values = val.find_all(exp.Tuple)

    return pl.DataFrame(
        {
            col: [raw.this for raw in raw_val.find_all(exp.Literal)]
            for col, raw_val in zip(val.named_selects, raw_values)
        }
    )


def request_for_sql(
    query: str, store: ContractStore
) -> list[tuple[RetrievalRequest, exp.Where | None]]:
    expr = parse_one(query)
    select_expr = expr.find_all(exp.Select)

    tables = set()
    table_alias: dict[str, str] = {}
    table_columns: dict[str, set[str]] = defaultdict(set)
    table_filters: dict[str, exp.Where] = {}
    unique_column_table_lookup: dict[str, str] = {}

    all_table_columns = {
        table_name: set(view.request_all.needed_requests[0].all_returned_columns)
        for table_name, view in store.feature_views.items()
    }
    all_model_columns = {
        table_name: set(model.predictions_view.request(table_name).all_returned_columns)
        for table_name, model in store.models.items()
    }

    for expr in select_expr:
        table_exps = list(expr.find_all(exp.Table))

        if len(table_exps) == 1:
            filter_exp = expr.find(exp.Where)
            if filter_exp:
                table_filters[table_exps[0].name] = filter_exp

        for table in table_exps:
            tables.add(table.name)
            table_alias[table.alias_or_name] = table.name

            for column in all_table_columns.get(table.name, set()).union(
                all_model_columns.get(table.name, set())
            ):
                if column in unique_column_table_lookup:
                    del unique_column_table_lookup[column]
                else:
                    unique_column_table_lookup[column] = table.name

        if expr.find(exp.Star):
            for table in tables:
                table_columns[table].update(
                    all_table_columns.get(table, set()).union(
                        all_model_columns.get(table, set())
                    )
                )
        else:
            for column in expr.find_all(exp.Column):
                source_table = table_alias.get(column.table)

                if source_table:
                    table_columns[source_table].add(column.name)
                    continue

                if column.table == "" and column.name in unique_column_table_lookup:
                    table_columns[unique_column_table_lookup[column.name]].add(
                        column.name
                    )
                    continue

                raise ValueError(
                    f"Unable to find table `{column.table}` for query `{query}`"
                )

    all_features: set[str] = set()

    for table, columns in table_columns.items():
        all_features.update(f"{table}:{column}" for column in columns)

    if not all_features:
        return []

    feature_request = store.requests_for_features(all_features)

    return [
        (req, table_filters.get(req.location.name))
        for req in feature_request.needed_requests
    ]
