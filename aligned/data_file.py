from __future__ import annotations

import polars as pl
from aligned.lazy_imports import pandas as pd


def upsert_on_column(columns: list[str], new_data: pl.LazyFrame, existing_data: pl.LazyFrame) -> pl.LazyFrame:
    new_data_columns = new_data.collect_schema().names()
    existing_data_columns = existing_data.collect_schema().names()

    column_diff = set(new_data_columns).difference(existing_data_columns)

    if column_diff:
        raise ValueError(f'Mismatching columns, missing columns {column_diff}.')

    combined = pl.concat([new_data, existing_data.select(new_data.columns)], how='vertical_relaxed')
    return combined.unique(columns, keep='first')


class DataFileReference:
    """
    A reference to a data file.

    It can therefore be loaded in and writen to.
    Either as a pandas data frame, or a dask data frame.
    """

    async def read_pandas(self) -> pd.DataFrame:
        raise NotImplementedError()

    async def to_pandas(self) -> pd.DataFrame:
        return await self.read_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        raise NotImplementedError()

    async def to_polars(self) -> pl.DataFrame:
        return (await self.to_lazy_polars()).collect()

    async def write_polars(self, df: pl.LazyFrame) -> None:
        raise NotImplementedError()

    async def write_pandas(self, df: pd.DataFrame) -> None:
        raise NotImplementedError()
