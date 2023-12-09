import pandas as pd
import polars as pl

from aligned.schemas.feature import Feature


class Validator:
    def validate_pandas(self, features: list[Feature], df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(type(self))

    def validate_polars(self, features: list[Feature], df: pl.LazyFrame) -> pl.LazyFrame:
        raise NotImplementedError(type(self))
