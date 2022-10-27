import logging
from dataclasses import dataclass, field

from redis.asyncio import Redis, StrictRedis  # type: ignore

from aligned.data_source.stream_data_source import StreamDataSource
from aligned.feature_source import FeatureSource, WritableFeatureSource
from aligned.online_source import OnlineSource
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.retrival_job import RetrivalJob
from aligned.schemas.codable import Codable
from aligned.schemas.feature import FeatureType
from aligned.schemas.feature_view import CompiledFeatureView

logger = logging.getLogger(__name__)


redis_manager: dict[str, StrictRedis] = {}


@dataclass
class RedisConfig(Codable):

    env_var: str

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

    def redis(self) -> Redis:
        if self.env_var not in redis_manager:
            redis_manager[self.env_var] = StrictRedis.from_url(self.url, decode_responses=True)

        return redis_manager[self.env_var]

    def online_source(self) -> 'RedisOnlineSource':
        return RedisOnlineSource(config=self)

    def stream_source(self, topic_name: str) -> 'RedisStreamSource':
        return RedisStreamSource(topic_name=topic_name, config=self)


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
        from aligned.redis.job import FactualRedisJob

        return FactualRedisJob(self.config, request.needed_requests, facts)

    async def write(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:

        redis = self.config.redis()
        data = await job.to_df()

        async with redis.pipeline(transaction=True) as pipe:

            written_count = 0
            # Run one query per row
            for request in requests:
                entity_ids = (
                    request.feature_view_name
                    + ':'
                    + data[sorted(request.entity_names)].astype('string').sum(axis=1).astype('string')
                )

                for feature in request.all_features:
                    mask = ~data[feature.name].isnull()

                    if feature.dtype == FeatureType('').bool:
                        mask = mask | ~data[feature.name].isna()

                    if mask.empty or (~mask).all():
                        continue

                    redis_values = data.loc[mask, feature.name].copy()

                    if feature.dtype == FeatureType('').bool:
                        # Redis do not support bools
                        redis_values = redis_values.astype(int)
                    elif feature.dtype == FeatureType('').datetime:
                        redis_values = redis_values.astype('int64') // 10**9

                    redis_values = redis_values.astype(str)

                    feature_key = entity_ids.loc[mask] + ':' + feature.name
                    for keys, value in zip(feature_key.values, redis_values.values):
                        pipe.set(keys, value)
                    written_count += 1

            await pipe.execute()


@dataclass
class RedisStreamSource(StreamDataSource):

    topic_name: str
    config: RedisConfig

    mappings: dict[str, str] = field(default_factory=dict)

    name: str = 'redis'

    def map_values(self, mappings: dict[str, str]) -> 'RedisStreamSource':
        return RedisStreamSource(
            topic_name=self.topic_name, config=self.config, mappings=self.mappings | mappings
        )
