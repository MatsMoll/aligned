import pandas as pd


class DataFileReference:
    """
    A reference to a data file.

    It can therefore be loaded in and writen to.
    Either as a pandas data frame, or a dask data frame.
    """

    async def read_pandas(self) -> pd.DataFrame:
        raise NotImplementedError()

    async def read_dask(self) -> pd.DataFrame:
        raise NotImplementedError()

    async def to_df(self) -> pd.DataFrame:
        await self.read_pandas()

    async def to_dask(self) -> pd.DataFrame:
        await self.read_dask()

    async def write_pandas(self, df: pd.DataFrame) -> None:
        raise NotImplementedError()

    async def write_dask(self, df: pd.DataFrame) -> None:
        raise NotImplementedError()
