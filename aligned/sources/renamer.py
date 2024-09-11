from __future__ import annotations

import polars as pl
from aligned.lazy_imports import pandas as pd


class Renamer:
    def rename_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        raise NotImplementedError(type(self))

    def rename_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(type(self))


class NoopRenamer(Renamer):
    def rename_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    def rename_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def camel_to_snake_case(column: str) -> str:
    return ''.join(['_' + char.lower() if char.isupper() else char for char in column]).lstrip('_')


class CamelToSnakeCase(Renamer):
    """
    Renames the colums from camel case to snake case
    """

    def rename_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.rename(camel_to_snake_case)

    def rename_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
