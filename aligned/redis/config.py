import logging
from dataclasses import dataclass, field

import polars as pl

try:
    from redis.asyncio import ConnectionPool, Redis, StrictRedis  # type: ignore
except ModuleNotFoundError:

    class Redis:  # type: ignore
        pass

    StrictRedis = Redis
    ConnectionPool = Redis

from aligned.data_source.stream_data_source import SinkableDataSource, StreamDataSource
from aligned.feature_source import FeatureSource, WritableFeatureSource
from aligned.online_source import OnlineSource
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.retrival_job import RetrivalJob
from aligned.schemas.codable import Codable
from aligned.schemas.feature import Feature, FeatureType
from aligned.schemas.feature_view import CompiledFeatureView

logger = logging.getLogger(__name__)


redis_manager: dict[str, ConnectionPool] = {}


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

        if 'REDIS_URL' not in os.environ:
            os.environ['REDIS_URL'] = 'redis://localhost:6379'

        return RedisConfig(env_var='REDIS_URL')

    def redis(self) -> Redis:
        if self.env_var not in redis_manager:
            redis_manager[self.env_var] = ConnectionPool.from_url(self.url, decode_responses=True)

        return StrictRedis(connection_pool=redis_manager[self.env_var])

    def online_source(self) -> 'RedisOnlineSource':
        return RedisOnlineSource(config=self)

    def stream(self, topic: str) -> 'RedisStreamSource':
        return RedisStreamSource(topic_name=topic, config=self)


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

    def features_for(self, facts: RetrivalJob, request: FeatureRequest) -> RetrivalJob:
        from aligned.redis.job import FactualRedisJob

        return FactualRedisJob(self.config, request.needed_requests, facts)

    async def write(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:

        redis = self.config.redis()
        data = await job.to_polars()

        async with redis.pipeline(transaction=True) as pipe:

            for request in requests:
                # Run one query per row
                filter_entity_query: pl.Expr = pl.lit(True)
                for entity_name in request.entity_names:
                    filter_entity_query = filter_entity_query & (pl.col(entity_name).is_not_null())

                request_data = data.filter(filter_entity_query).with_column(
                    (
                        pl.lit(request.location.identifier)
                        + pl.lit(':')
                        + pl.concat_str(sorted(request.entity_names))
                    ).alias('id')
                )

                features = ['id']

                for feature in request.returned_features:

                    expr = pl.col(feature.name).cast(pl.Utf8).alias(feature.name)

                    if feature.dtype == FeatureType('').bool:
                        # Redis do not support bools
                        expr = (
                            pl.col(feature.name).cast(pl.Int8, strict=False).cast(pl.Utf8).alias(feature.name)
                        )
                    elif feature.dtype == FeatureType('').datetime:
                        expr = pl.col(feature.name).dt.timestamp('ms').cast(pl.Utf8).alias(feature.name)
                    elif feature.dtype == FeatureType('').embedding or feature.dtype == FeatureType('').array:
                        import json

                        expr = pl.col(feature.name).apply(lambda x: json.dumps(x.to_list()))

                    request_data = request_data.with_column(expr)
                    features.append(feature.name)

                redis_frame = request_data.select(features).collect()

                for record in redis_frame.to_dicts():
                    pipe.hset(record['id'], mapping={key: value for key, value in record.items() if value})
                    for key, value in record.items():
                        if value is None:
                            logger.info(f"Deleting {key} from {record['id']}")
                            pipe.hdel(record['id'], key)
                await pipe.execute()


@dataclass
class RedisStreamSource(StreamDataSource, SinkableDataSource):

    topic_name: str
    config: RedisConfig

    mappings: dict[str, str] = field(default_factory=dict)

    name: str = 'redis'

    def map_values(self, mappings: dict[str, str]) -> 'RedisStreamSource':
        return RedisStreamSource(
            topic_name=self.topic_name, config=self.config, mappings=self.mappings | mappings
        )

    def make_redis_friendly(self, data: pl.LazyFrame, features: set[Feature]) -> pl.LazyFrame:
        # Run one query per row
        for feature in features:

            expr = pl.col(feature.name)

            if feature.dtype == FeatureType('').bool:
                # Redis do not support bools
                expr = pl.col(feature.name).cast(pl.Int8, strict=False)
            elif feature.dtype == FeatureType('').datetime:
                expr = pl.col(feature.name).dt.timestamp('ms')
            elif feature.dtype == FeatureType('').embedding or feature.dtype == FeatureType('').array:
                import json

                expr = pl.col(feature.name).apply(lambda x: json.dumps(x.to_list()))

            data = data.with_column(expr.cast(pl.Utf8).alias(feature.name))

        return data

    async def write_to_stream(self, job: RetrivalJob) -> None:
        redis = self.config.redis()
        df = await job.to_polars()

        df = self.make_redis_friendly(df, job.request_result.features.union(job.request_result.entities))
        values = df.collect()

        async with redis.pipeline(transaction=True) as pipe:
            _ = [
                pipe.xadd(
                    self.topic_name,
                    {key: value for key, value in record.items() if value},
                )
                for record in values.to_dicts()
            ]
            await pipe.execute()
