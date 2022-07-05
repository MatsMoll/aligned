from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from mashumaro.types import SerializableType
from pandas import DataFrame

from aladdin.codable import Codable
from aladdin.redis.config import RedisConfig

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

    def lock(self, lock_name: str, redis_config: RedisConfig) -> Enricher:
        return RedisLockEnricher(lock_name=lock_name, enricher=self, config=redis_config)

    def cache(self, ttl: timedelta, cache_key: str) -> Enricher:
        return FileCacheEnricher(ttl, f'./cache/{cache_key}', self)

    @abstractmethod
    async def load(self) -> DataFrame:
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
    name: str = 'redis_lock'

    def __init__(self, lock_name: str, enricher: Enricher, config: RedisConfig):
        self.lock_name = lock_name
        self.config = config
        self.enricher = enricher

    async def load(self) -> DataFrame:
        async with self.config.redis().lock(self.lock_name) as _:
            return await self.enricher.load()


@dataclass
class FileEnricher(Enricher):

    file: Path
    name: str = 'file'

    async def load(self) -> DataFrame:
        import pandas as pd

        if self.file.suffix == '.csv':
            return pd.read_csv(self.file.absolute())
        else:
            return pd.read_parquet(self.file.absolute())


@dataclass
class FileStatEnricher(Enricher):

    stat: str
    columns: list[str]
    enricher: Enricher

    async def load(self) -> DataFrame:
        data = await self.enricher.load()
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

    async def load(self) -> DataFrame:
        should_load_source = False
        file_uri = Path(self.file_path).absolute()
        try:
            # Checks last modified metadata field
            modified_at = datetime.fromtimestamp(file_uri.stat().st_mtime)
            compare = datetime.now() - self.ttl
            should_load_source = modified_at < compare
        except FileNotFoundError:
            should_load_source = True

        if should_load_source:
            logger.info('Fetching from source')
            data: DataFrame = await self.enricher.load()
            file_uri.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f'Storing cache at file {file_uri.as_uri()}')
            data.to_parquet(file_uri)
        else:
            import pandas as pd

            logger.info('Loading cache')

            data = pd.read_parquet(file_uri)
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

    async def load(self) -> DataFrame:
        import os

        from databases import Database

        async with Database(os.environ[self.url_env]) as db:
            records = await db.fetch_all(self.query, values=self.values)
        df = DataFrame.from_records([dict(record) for record in records])
        for name, dtype in df.dtypes.iteritems():
            if dtype == 'object':  # Need to convert the databases UUID type
                df[name] = df[name].astype('str')
        return df
