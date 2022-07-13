import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from redis.asyncio import Redis, StrictRedis  # type: ignore

from aladdin.codable import Codable
from aladdin.feature import FeatureType
from aladdin.feature_source import FeatureSource, WritableFeatureSource
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.online_source import OnlineSource
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest
from aladdin.retrival_job import RetrivalJob

logger = logging.getLogger(__name__)


class RedisEvictionPolicy(Enum):
    NO_EVICTION = 'noeviction'
    ALL_KEYS_LRU = 'allkeys-lru'
    ALL_KEYS_LFU = 'allkeys-lfu'
    ALL_KEYS_RANDOM = 'allkeys-random'

    VOLATILE_LRU = 'volatile-lru'
    VOLATILE_FRU = 'volatile-fru'
    VOLATILE_RANDOM = 'volatile-random'
    VOLATILE_TTL = 'volatile-ttl'


@dataclass
class RedisConfig(Codable):

    env_var: str
    _eviction_policy: RedisEvictionPolicy = RedisEvictionPolicy.NO_EVICTION

    @property
    def url(self) -> str:
        import os

        return os.environ[self.env_var]

    @staticmethod
    def from_url(url: str) -> 'RedisConfig':
        import os

        os.environ['REDIS_URL'] = url
        return RedisConfig(env_var='REDIS_URL')

    @staticmethod
    def localhost() -> 'RedisConfig':
        import os

        os.environ['REDIS_URL'] = 'redis://localhost:6379'
        return RedisConfig(env_var='REDIS_URL')

    def eviction_policy(self, policy: RedisEvictionPolicy) -> 'RedisConfig':
        self._eviction_policy = policy
        return self

    async def redis(self) -> Redis:
        global _redis
        try:
            return _redis  # type: ignore
        except NameError:
            _redis = StrictRedis.from_url(self.url, decode_responses=True)  # type: ignore
            await _redis.config_set('maxmemory-policy', self._eviction_policy.value)  # type: ignore
            return _redis  # type: ignore

    def online_source(self) -> 'RedisOnlineSource':
        return RedisOnlineSource(config=self)


@dataclass
class RedisOnlineSource(OnlineSource):

    config: RedisConfig
    source_type = 'redis'

    def feature_source(self, feature_views: set[CompiledFeatureView]) -> 'RedisSource':
        return RedisSource(self.config)


@dataclass
class RedisSource(FeatureSource, WritableFeatureSource):

    config: RedisConfig
    batch_size = 1_000_000

    def all_for(self, request: FeatureRequest, limit: int | None = None) -> RetrivalJob:
        raise NotImplementedError()

    def features_for(self, facts: dict[str, list], request: FeatureRequest) -> RetrivalJob:
        from aladdin.redis.job import FactualRedisJob

        return FactualRedisJob(self.config, request.needed_requests, facts)

    async def write(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:
        from aladdin.redis.job import key

        redis = await self.config.redis()
        data = await job.to_df()
        logger.info(f'Writing {data.shape} features to redis')
        async with redis.pipeline(transaction=True) as pipe:

            written_count = 0
            for _, row in data.iterrows():
                # Run one query per row
                for request in requests:
                    entity_ids = row[list(request.entity_names)]
                    entity_id = ':'.join([str(entity_id) for entity_id in entity_ids])
                    if not entity_id:
                        continue

                    for feature in request.all_features:
                        value = row[feature.name]
                        if value and not pd.isnull(value):
                            if feature.dtype == FeatureType('').bool:
                                # Redis do not support bools
                                value = int(value)
                            elif feature.dtype == FeatureType('').datetime:
                                value = value.timestamp()

                            string_value = f'{value}'

                            feature_key = key(request, entity_id, feature)
                            pipe.set(feature_key, string_value)
                            written_count += 1

                if written_count % self.batch_size == 0:
                    await pipe.execute()
                    logger.info(f'Written {written_count} rows')
            await pipe.execute()
