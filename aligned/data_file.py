import pandas as pd
import polars as pl


class DataFileReference:
    """
    A reference to a data file.

    It can therefore be loaded in and writen to.
    Either as a pandas data frame, or a dask data frame.
    """

    async def read_pandas(self) -> pd.DataFrame:
        raise NotImplementedError()

    async def to_pandas(self) -> pd.DataFrame:
        await self.read_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        raise NotImplementedError()

    async def write_polars(self, df: pl.LazyFrame) -> None:
        raise NotImplementedError()

    async def write_pandas(self, df: pd.DataFrame) -> None:
        raise NotImplementedError()
