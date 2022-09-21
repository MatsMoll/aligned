from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from mashumaro.types import SerializableType

from aladdin.codable import Codable
from aladdin.redis.config import RedisConfig

with suppress(ModuleNotFoundError):
    import dask.dataframe as dd

logger = logging.getLogger(__name__)


class StatisticEricher:
    def std(self, columns: set[str]) -> Enricher:
        raise NotImplementedError()

    def mean(self, columns: set[str]) -> Enricher:
        raise NotImplementedError()


class Enricher(ABC, Codable, SerializableType):

    name: str

    def _serialize(self) -> dict:
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> Enricher:
        name_type = value['name']
        del value['name']
        data_class = SupportedEnrichers.shared().types[name_type]
        return data_class.from_dict(value)

    def lock(self, lock_name: str, redis_config: RedisConfig, timeout: int = 60) -> Enricher:
        return RedisLockEnricher(lock_name=lock_name, enricher=self, config=redis_config, timeout=timeout)

    def cache(self, ttl: timedelta, cache_key: str) -> Enricher:
        return FileCacheEnricher(ttl, f'./cache/{cache_key}', self)

    @abstractmethod
    async def load(self) -> pd.DataFrame:
        pass

    @abstractmethod
    async def as_dask(self) -> dd.DataFrame:
        pass


class SupportedEnrichers:

    types: dict[str, type[Enricher]]

    _shared: SupportedEnrichers | None = None

    def __init__(self) -> None:
        self.types = {}

        default_types: list[type[Enricher]] = [RedisLockEnricher, FileCacheEnricher, SqlDatabaseEnricher]
        for enrich_type in default_types:
            self.add(enrich_type)

    def add(self, enrich_type: type[Enricher]) -> None:
        self.types[enrich_type.name] = enrich_type

    @classmethod
    def shared(cls) -> SupportedEnrichers:
        if cls._shared:
            return cls._shared
        cls._shared = SupportedEnrichers()
        return cls._shared


@dataclass
class RedisLockEnricher(Enricher):

    enricher: Enricher
    config: RedisConfig
    lock_name: str
    timeout: int
    name: str = 'redis_lock'

    def __init__(self, lock_name: str, enricher: Enricher, config: RedisConfig, timeout: int):
        self.lock_name = lock_name
        self.config = config
        self.enricher = enricher
        self.timeout = timeout

    async def load(self) -> pd.DataFrame:
        redis = self.config.redis()
        async with redis.lock(self.lock_name, timeout=self.timeout) as _:
            return await self.enricher.load()

    async def as_dask(self) -> dd.DataFrame:
        redis = self.config.redis()
        async with redis.lock(self.lock_name, timeout=self.timeout) as _:
            return await self.enricher.as_dask()


@dataclass
class CsvFileEnricher(Enricher):

    file: Path
    name: str = 'file'

    async def load(self) -> pd.DataFrame:

        if self.file.suffix == '.csv':
            return pd.read_csv(self.file.absolute())
        else:
            return pd.read_parquet(self.file.absolute())

    async def as_dask(self) -> dd.DataFrame:
        if self.file.suffix == '.csv':
            return dd.read_csv(self.file.absolute())
        else:
            return dd.read_parquet(self.file.absolute())


@dataclass
class LoadedStatEnricher(Enricher):

    stat: str
    columns: list[str]
    enricher: Enricher

    async def load(self) -> pd.DataFrame:
        data = await self.enricher.load()
        if self.stat == 'mean':
            return data[self.columns].mean()
        elif self.stat == 'std':
            return data[self.columns].std()
        else:
            raise ValueError(f'Not supporting stat: {self.stat}')

    async def as_dask(self) -> dd.DataFrame:
        data = await self.enricher.as_dask()
        if self.stat == 'mean':
            return data[self.columns].mean()
        elif self.stat == 'std':
            return data[self.columns].std()
        else:
            raise ValueError(f'Not supporting stat: {self.stat}')


@dataclass
class FileCacheEnricher(Enricher):

    ttl: timedelta
    file_path: str
    enricher: Enricher
    name: str = 'file_cache'

    def is_out_of_date_cache(self) -> bool:
        file_uri = Path(self.file_path).absolute()
        try:
            # Checks last modified metadata field
            modified_at = datetime.fromtimestamp(file_uri.stat().st_mtime)
            compare = datetime.now() - self.ttl
            return modified_at < compare
        except FileNotFoundError:
            return True

    async def load(self) -> pd.DataFrame:
        file_uri = Path(self.file_path).absolute()

        if self.is_out_of_date_cache():
            logger.info('Fetching from source')
            data: pd.DataFrame = await self.enricher.load()
            file_uri.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f'Storing cache at file {file_uri.as_uri()}')
            data.to_parquet(file_uri)
        else:
            logger.info('Loading cache')
            data = pd.read_parquet(file_uri)
        return data

    async def as_dask(self) -> dd.DataFrame:
        file_uri = Path(self.file_path).absolute()

        if self.is_out_of_date_cache():
            logger.info('Fetching from source')
            data: dd.DataFrame = await self.enricher.as_dask()
            file_uri.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f'Storing cache at file {file_uri.as_uri()}')
            data.to_parquet(file_uri)
        else:
            logger.info('Loading cache')
            data = dd.read_parquet(file_uri)
        return data


@dataclass
class SqlDatabaseEnricher(Enricher):

    query: str
    values: dict | None
    url_env: str
    name: str = 'sql'

    def __init__(self, url_env: str, query: str, values: dict | None = None) -> None:
        self.query = query
        self.values = values
        self.url_env = url_env

    async def load(self) -> pd.DataFrame:
        import os

        from databases import Database

        async with Database(os.environ[self.url_env]) as db:
            records = await db.fetch_all(self.query, values=self.values)
        df = pd.DataFrame.from_records([dict(record) for record in records])
        for name, dtype in df.dtypes.iteritems():
            if dtype == 'object':  # Need to convert the databases UUID type
                df[name] = df[name].astype('str')
        return df

    async def as_dask(self) -> dd.DataFrame:
        pdf = await self.load()
        return dd.from_pandas(pdf)
