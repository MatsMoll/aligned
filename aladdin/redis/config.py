from redis.asyncio import Redis
from dataclasses import dataclass
from aladdin.codable import Codable

from aladdin.online_source import OnlineSource
from aladdin.feature_source import FeatureSource

from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.retrival_job import RetrivalJob
from aladdin.request.retrival_request import RetrivalRequest
import pandas as pd

@dataclass
class RedisConfig(Codable):

    env_var: str

    @property
    def url(self) -> str:
        import os
        return os.environ[self.env_var]

    @staticmethod
    def from_url(url: str) -> "RedisConfig":
        import os
        os.environ["REDIS_URL"] = url
        return RedisConfig(env_var="REDIS_URL")

    @staticmethod
    def localhost() -> "RedisConfig":
        import os
        os.environ["REDIS_URL"] = "redis://localhost:6379"
        return RedisConfig(env_var="REDIS_URL")

    def redis(self) -> "Redis":
        global _redis
        try:
            return _redis
        except NameError:
            _redis = Redis.from_url(self.url)
            return _redis

    def online_source(self) -> "RedisOnlineSource":
        return RedisOnlineSource(config=self)

@dataclass
class RedisOnlineSource(OnlineSource):

    config: RedisConfig
    source_type = "redis"

    def feature_source(self, feature_views: set[CompiledFeatureView]) -> "RedisSource":
        return RedisSource(self.config)

@dataclass
class RedisSource(FeatureSource):

    config: RedisConfig

    async def store(self, job: RetrivalJob, requests: set[RetrivalRequest]) -> None:
        from aladdin.redis.job import key

        redis = self.config.redis()
        data = await job.to_df()

        async with redis.pipeline(transaction=True) as pipe:
            for _, row in data.iterrows():
                # Run one query per row
                for request in requests:
                    entity_ids = row[request.entity_names]
                    entity_id = ":".join([str(entity_id) for entity_id in entity_ids])
                    if not entity_id:
                        continue

                    for feature in request.all_features:
                        value = row[feature.name]
                        if value and not pd.isnull(value):
                            if feature.dtype == bool:
                                # Redis do not support bools
                                value = int(value)
                            feature_key = key(request, entity_id, feature)
                            pipe.set(feature_key, value)

            await pipe.execute()


    def features_for(self, facts: dict[str, list], requests: set[RetrivalRequest]) -> RetrivalJob:
        from aladdin.redis.job import FactualRedisJob
        return FactualRedisJob(self.config, requests, facts)