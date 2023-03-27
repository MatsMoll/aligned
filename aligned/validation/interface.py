import pandas as pd
import polars as pl

from aligned.schemas.feature import Feature


class Validator:
    async def validate_pandas(self, features: list[Feature], df: pd.DataFrame) -> pd.DataFrame:
        pass

    async def validate_polars(self, features: list[Feature], df: pl.LazyFrame) -> pl.LazyFrame:
        pass
