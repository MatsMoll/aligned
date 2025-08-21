import polars as pl
from aligned.polars_to_spark import polars_expression_to_spark


def test_basic_expressions() -> None:
    test_cases = [
        (pl.col("a") == 10, "(a = 10)"),
        (pl.col("a") == "test", "(a = 'test')"),
        ((pl.col("a") * 100) + pl.col("other") < 10, "(((a * 100) + other) < 10)"),
    ]

    for expr, spark_sql in test_cases:
        output = polars_expression_to_spark(expr)
        assert (
            output == spark_sql
        ), f"Expected '{spark_sql}' but got '{output}' from {expr}"
