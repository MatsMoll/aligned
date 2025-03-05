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
    return "".join(
        ["_" + char.lower() if char.isupper() else char for char in column]
    ).lstrip("_")


def snake_to_pascal(column: str) -> str:
    return "".join(
        [s[0].upper() + s[1:].lower() if s else s for s in column.split("_")]
    )


class CamelToSnakeCase(Renamer):
    """
    Renames the columns from camel case to snake case
    """

    def rename_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.rename(camel_to_snake_case)


class SnakeToPascalCase(Renamer):
    """
    Renames the columns from snake case to pascal case
    """

    def rename_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return df.rename(snake_to_pascal)
