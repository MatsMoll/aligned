import polars as pl
from aligned.polars_to_spark import ExpressionNode


def test_polars_decoding() -> None:
    test_cases = [
        # Basic comparisons
        pl.col("a") == False,  # noqa: E712
        pl.col("a") == 10,
        pl.col("a") == "test",
        pl.col("a") != 5,
        pl.col("a") > 10,
        pl.col("a") < 5,
        pl.col("a") >= 10,
        pl.col("a") <= 5,
        # Arithmetic operations
        (pl.col("a") * 100) + pl.col("other") < 10,
        pl.col("a") - pl.col("b"),
        pl.col("a") / pl.col("b"),
        pl.col("a") % 3,
        pl.col("a") ** 2,
        pl.col("a") ** 3,
        # Logical operations
        (pl.col("a") > 5) & (pl.col("b") < 10),
        (pl.col("a") > 5) | (pl.col("b") < 10),
        ~(pl.col("a") > 5),
        # Null/NaN checks
        pl.col("a").is_null(),
        pl.col("a").is_not_null(),
        pl.col("a").is_nan(),
        pl.col("a").is_not_nan(),
        # String operations
        pl.col("name").str.contains("test"),
        pl.col("name").str.starts_with("prefix"),
        pl.col("name").str.ends_with("suffix"),
        pl.col("name").str.len_chars(),
        pl.col("name").str.to_lowercase(),
        pl.col("name").str.to_uppercase(),
        # Mathematical functions
        pl.col("a").exp(),
        pl.col("a").log(),
        pl.col("a").log10(),
        pl.col("a").sqrt(),
        pl.col("a").abs(),
        pl.col("a").sin(),
        pl.col("a").cos(),
        pl.col("a").tan(),
        # Range and conditional checks
        pl.col("a").is_between(1, 4),
        # Date/time operations (if applicable)
        pl.col("date").dt.year(),
        pl.col("date").dt.month(),
        pl.col("date").dt.day(),
        pl.col("date").dt.iso_year(),
        # Type casting
        pl.col("a").cast(pl.Int64),
        pl.col("a").cast(pl.Float64),
        pl.col("a").cast(pl.Utf8),
    ]

    for expr in test_cases:
        content = expr.meta.serialize(format="json")
        try:
            node = ExpressionNode.model_validate_json(content)
            _ = node.to_expression()
        except Exception as e:
            print(expr)
            print(content)
            raise e
