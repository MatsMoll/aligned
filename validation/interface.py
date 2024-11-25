from __future__ import annotations
import polars as pl

from aligned.lazy_imports import pandas as pd
from aligned.schemas.feature import Feature


class Validator:
    def validate_pandas(self, features: list[Feature], df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(type(self))

    def validate_polars(self, features: list[Feature], df: pl.LazyFrame) -> pl.LazyFrame:
        raise NotImplementedError(type(self))


class PolarsValidator(Validator):
    def validate_polars(self, features: list[Feature], df: pl.LazyFrame) -> pl.LazyFrame:
        from aligned.retrival_job import polars_filter_expressions_from

        raw_exprs = polars_filter_expressions_from(features)
        expressions = [expr for expr, _ in raw_exprs]

        return df.filter(pl.all_horizontal(expressions))

    def validate_pandas(self, features: list[Feature], df: pd.DataFrame) -> pd.DataFrame:
        return self.validate_polars(features, pl.from_pandas(df).lazy()).collect().to_pandas()
