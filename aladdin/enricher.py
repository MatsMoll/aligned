from abc import ABC, abstractmethod
from pandas import DataFrame
from datetime import timedelta, datetime
from pathlib import Path
from dataclasses import dataclass
from mashumaro.types import SerializableType
from typing import Optional

from aladdin.redis.config import RedisConfig


class Enricher(ABC, SerializableType):

    name: str

    def _serialize(self):
        return self.to_dict()
    
    @classmethod
    def _deserialize(cls, value: dict[str]) -> 'Enricher':
        name_type = value["name"]
        del value["name"]
        data_class = SupportedEnrichers.shared().types[name_type]
        return data_class.from_dict(value)

    def lock(self, lock_name: str, redis_config: RedisConfig) -> 'Enricher':
        return RedisLockEnricher(lock_name=lock_name, config=redis_config)

    def cache(self, ttl: timedelta, cache_key: str) -> 'Enricher':
        return FileCacheEnricher(ttl, Path(f"./cache/{cache_key}"), self)

    @abstractmethod
    async def load(self) -> DataFrame:
        pass

class SupportedEnrichers:

    types: dict[str, type[Enricher]]

    _shared: Optional["SupportedEnrichers"] = None

    def __init__(self):
        self.types = {}
        
        for enrich_type in [
            RedisLockEnricher,
            FileCacheEnricher,
            SqlDatabaseEnricher
        ]:
            self.add(enrich_type)

    def add(self, transformation: type[Enricher]):
        self.types[transformation.name] = transformation

    @classmethod
    def shared(cls) -> "SupportedEnrichers":
        if cls._shared:
            return cls._shared
        cls._shared = SupportedEnrichers()
        return cls._shared


@dataclass
class RedisLockEnricher(Enricher):

    enricher: Enricher
    config: RedisConfig
    lock_name: str
    name: str = "redis_lock"
    
    def __init__(self, lock_name: str, enricher: Enricher, config: RedisConfig):
        self.lock_name = lock_name
        self.config = config
        self.enricher = enricher

    async def load(self) -> DataFrame:
        async with self.config.redis().lock(self.lock_name) as _:
            return await self.enricher.load()

@dataclass
class FileCacheEnricher(Enricher):

    ttl: timedelta
    file: Path
    enricher: Enricher
    name: str = "file_cache"

    async def load(self):
        import os
        should_load = False
        file_uri = self.file.absolute()
        try:
            # Checks last modified metadata field
            modified_at = datetime.fromtimestamp(os.stat(file_uri).st_mtime)
            should_load = modified_at < datetime.now() - self.ttl
        except FileNotFoundError:
            should_load = True
        
        if should_load:
            data: DataFrame = await self.enricher.load()
            data.to_parquet(file_uri)
        else:
            import pandas as pd
            data = pd.read_parquet(file_uri)
        return data


@dataclass
class SqlDatabaseEnricher(Enricher):
    
    query: str
    values: dict | None
    url: str
    name: str = "sql"

    def __init__(self, url: str, query: str, values: dict | None = None) -> None:
        self.query = query
        self.values = values
        self.url = url

    async def load(self) -> DataFrame:
        from databases import Database
        
        async with Database(self.url) as db:
            records = await db.fetch_all(self.query, values=self.values)
        df = DataFrame.from_records([dict(record) for record in records])
        for name, dtype in df.dtypes.iteritems():
            if "object" == dtype: # Need to convert the databases UUID type
                df[name] = df[name].astype("str")
        return df