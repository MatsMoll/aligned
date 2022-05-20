from abc import ABC, abstractmethod
from pandas import DataFrame
from redis.asyncio import Redis
from datetime import timedelta, datetime
from pathlib import Path
from dataclasses import dataclass


class Enricher(ABC):

    @abstractmethod
    async def load(self) -> DataFrame:
        pass

@dataclass
class RedisLockEnricher(Enricher):

    enricher: Enricher
    redis: Redis
    lock_name: str
    
    def __init__(self, lock_name: str, enricher: Enricher, redis: Redis):
        self.lock_name = lock_name
        self.redis = redis
        self.enricher = enricher

    async def load(self) -> DataFrame:
        async with self.redis.lock(self.lock_name) as _:
            return await self.enricher.load()

@dataclass
class FileCacheEnricher(Enricher):

    ttl: timedelta
    file: Path
    enricher: Enricher

    async def load(self):
        import os
        should_reload = False
        file_uri = self.file.absolute()
        try:
            # Checks last modified metadata field
            modified_at = datetime.fromtimestamp(os.stat(file_uri).st_mtime)
            should_reload = modified_at < datetime.now() - self.ttl
        except FileNotFoundError:
            should_reload = True
        
        if should_reload:
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