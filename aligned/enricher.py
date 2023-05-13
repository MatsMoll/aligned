from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable
from aligned.sources.redis import RedisConfig

logger = logging.getLogger(__name__)


@dataclass
class TimespanSelector(Codable):
    timespand: timedelta
    time_column: str


class StatisticEricher:
    def std(
        self, columns: set[str], time: TimespanSelector | None = None, limit: int | None = None
    ) -> Enricher:
        raise NotImplementedError()

    def mean(
        self, columns: set[str], time: TimespanSelector | None = None, limit: int | None = None
    ) -> Enricher:
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
        return FileCacheEnricher(ttl, cache_key, self)

    @abstractmethod
    async def as_df(self) -> pd.DataFrame:
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

    async def as_df(self) -> pd.DataFrame:
        redis = self.config.redis()
        async with redis.lock(self.lock_name, timeout=self.timeout) as _:
            return await self.enricher.as_df()


@dataclass
class CsvFileSelectedEnricher(Enricher):
    file: str
    time: TimespanSelector | None = field(default=None)
    limit: int | None = field(default=None)
    name: str = 'selective_file'

    async def as_df(self) -> pd.DataFrame:
        dates_to_parse = None
        if self.time:
            dates_to_parse = [self.time.time_column]

        uri = self.file
        path = Path(self.file)
        if 'http' not in path.parts[0]:
            uri = str(path.absolute())

        if self.limit:
            file = pd.read_csv(uri, nrows=self.limit, parse_dates=dates_to_parse)
        else:
            file = pd.read_csv(uri, nrows=self.limit, parse_dates=dates_to_parse)

        if not self.time:
            return file

        date = datetime.now() - self.time.timespand
        selector = file[self.time.time_column] >= date
        return file.loc[selector]


@dataclass
class CsvFileEnricher(Enricher):

    file: str
    name: str = 'file'

    def selector(
        self, time: TimespanSelector | None = None, limit: int | None = None
    ) -> CsvFileSelectedEnricher:
        return CsvFileSelectedEnricher(self.file, time=time, limit=limit)

    async def as_df(self) -> pd.DataFrame:
        return pd.read_csv(self.file)


@dataclass
class LoadedStatEnricher(Enricher):

    stat: str
    columns: list[str]
    enricher: Enricher
    mapping_keys: dict[str, str] = field(default_factory=dict)

    async def as_df(self) -> pd.DataFrame:
        data = await self.enricher.as_df()
        renamed = data.rename(columns=self.mapping_keys)
        if self.stat == 'mean':
            return renamed[self.columns].mean()
        elif self.stat == 'std':
            return renamed[self.columns].std()
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

    async def as_df(self) -> pd.DataFrame:
        file_uri = Path(self.file_path).absolute()

        if self.is_out_of_date_cache():
            logger.info('Fetching from source')
            data: pd.DataFrame = await self.enricher.as_df()
            file_uri.parent.mkdir(exist_ok=True, parents=True)
            logger.info(f'Storing cache at file {file_uri.as_uri()}')
            data.to_parquet(file_uri)
        else:
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

    async def as_df(self) -> pd.DataFrame:
        import os

        import connectorx as cx

        df = cx.read_sql(os.environ[self.url_env], self.query, return_type='pandas')

        for name, dtype in df.dtypes.iteritems():
            if dtype == 'object':  # Need to convert the databases UUID type
                df[name] = df[name].astype('str')

        return df
