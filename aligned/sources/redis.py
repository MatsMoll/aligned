from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import polars as pl

from aligned.streams.interface import ReadableStream
from aligned.lazy_imports import redis
from aligned.data_source.batch_data_source import ColumnFeatureMappable
from aligned.data_source.stream_data_source import SinkableDataSource, StreamDataSource
from aligned.feature_source import (
    FeatureSource,
    FeatureSourceFactory,
    WritableFeatureSource,
)
from aligned.request.retrieval_request import FeatureRequest, RetrievalRequest
from aligned.retrieval_job import RetrievalJob
from aligned.schemas.codable import Codable
from aligned.schemas.feature import Feature, FeatureType
from aligned.schemas.record_coders import PassthroughRecordCoder, RecordCoder
from aligned.schemas.vector_storage import VectorIndex, VectorStorage

if TYPE_CHECKING:
    from redis.commands.search.field import (
        NumericField,
        TagField,
        TextField,
        VectorField,
    )


logger = logging.getLogger(__name__)


redis_manager: dict[str, redis.ConnectionPool] = {}


@dataclass
class RedisConfig(Codable, FeatureSourceFactory):
    env_var: str

    @property
    def url(self) -> str:
        import os

        return os.environ[self.env_var]

    @staticmethod
    def from_url(url: str) -> RedisConfig:
        import os

        os.environ["REDIS_URL"] = url
        return RedisConfig(env_var="REDIS_URL")

    @staticmethod
    def localhost() -> RedisConfig:
        import os

        if "REDIS_URL" not in os.environ:
            os.environ["REDIS_URL"] = "redis://localhost:6379"

        return RedisConfig(env_var="REDIS_URL")

    def feature_source(self) -> FeatureSource:
        return RedisSource(self)

    def redis(self) -> redis.Redis:
        if self.env_var not in redis_manager:
            redis_manager[self.env_var] = redis.ConnectionPool.from_url(
                self.url, decode_responses=True
            )

        return redis.StrictRedis(connection_pool=redis_manager[self.env_var])

    def stream(self, topic: str) -> RedisStreamSource:
        return RedisStreamSource(topic_name=topic, config=self)

    def index(
        self,
        name: str,
        initial_cap: int | None = None,
        distance_metric: str | None = None,
        algorithm: str | None = None,
        embedding_type: str | None = None,
    ) -> RedisVectorIndex:
        return RedisVectorIndex(
            config=self,
            name=name,
            initial_cap=initial_cap or 10000,
            distance_metric=distance_metric or "COSINE",
            index_alogrithm=algorithm or "FLAT",
            embedding_type=embedding_type or "FLOAT32",
        )


@dataclass
class RedisVectorIndex(VectorStorage):
    config: RedisConfig

    name: str
    initial_cap: int
    distance_metric: str
    index_alogrithm: str
    embedding_type: str

    type_name: str = "redis"

    async def create_index(self, index: VectorIndex) -> None:
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType  # type: ignore

        redis = self.config.redis()
        try:
            info = await redis.ft(self.name).info()
            logger.info(f"Index {self.name} already exists: {info}")
        except Exception:
            logger.info(
                f"Creating index {self.name} with prefix {index.location.identifier}..."
            )
            index_definition = IndexDefinition(
                prefix=[index.location.identifier], index_type=IndexType.HASH
            )
            fields = [
                self.index_schema(feature, index)
                for feature in set(index.metadata).union({index.vector})
            ]
            await redis.ft(self.name).create_index(
                fields=fields,
                definition=index_definition,
            )

    def index_schema(
        self, feature: Feature, index: VectorIndex
    ) -> Union[NumericField, TextField, TagField, VectorField]:
        from redis.commands.search.field import (
            NumericField,
            TagField,
            TextField,
            VectorField,
        )  # type: ignore

        if feature.dtype.is_numeric:
            return NumericField(name=feature.name)

        match feature.dtype.name:
            case "string":
                return TextField(name=feature.name)
            case "uuid":
                return TagField(name=feature.name)
            case "embedding":
                if feature.name != index.vector.name:
                    raise ValueError(
                        f"The metadata can not contain a feature embedding. like: {feature.name}"
                    )

                return VectorField(
                    name=feature.name,
                    algorithm=self.index_alogrithm,
                    attributes={
                        "TYPE": self.embedding_type,
                        "DIM": index.vector_dim,
                        "DISTANCE_METRIC": self.distance_metric,
                        "INITIAL_CAP": self.initial_cap,
                    },
                )
        raise ValueError(f"Unsupported feature type {feature.dtype.name}")

    async def upsert_polars(self, df: pl.LazyFrame, index: VectorIndex) -> None:
        logger.info(f"Upserting {len(df.columns)} into index {self.name}...")


@dataclass
class RedisSource(FeatureSource, WritableFeatureSource):
    config: RedisConfig
    batch_size = 1_000_000

    def all_for(
        self, request: FeatureRequest, limit: int | None = None
    ) -> RetrievalJob:
        raise NotImplementedError()

    def features_for(
        self, facts: RetrievalJob, request: FeatureRequest
    ) -> RetrievalJob:
        from aligned.redis.job import FactualRedisJob

        needed_requests = [
            req
            for req in request.needed_requests
            if req.location.location_type != "combined_view"
        ]
        combined = [
            req
            for req in request.needed_requests
            if req.location.location_type == "combined_view"
        ]
        return FactualRedisJob(self.config, needed_requests, facts).combined_features(
            combined
        )

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        redis = self.config.redis()
        data = await job.to_lazy_polars()

        async with redis.pipeline(transaction=True) as pipe:
            # Run one query per row
            filter_entity_query: pl.Expr = pl.lit(True)
            for entity_name in request.entity_names:
                filter_entity_query = filter_entity_query & (
                    pl.col(entity_name).is_not_null()
                )

            request_data = data.filter(filter_entity_query).with_columns(
                (
                    pl.concat_str(
                        [pl.lit(request.location.identifier), pl.lit(":")]
                        + [pl.col(col) for col in sorted(request.entity_names)]
                    )
                ).alias("id")
            )

            features = ["id"]

            for feature in request.returned_features:
                expr = pl.col(feature.name).cast(pl.Utf8).alias(feature.name)

                if feature.dtype == FeatureType.boolean():
                    # Redis do not support bools
                    expr = (
                        pl.col(feature.name)
                        .cast(pl.Int8, strict=False)
                        .cast(pl.Utf8)
                        .alias(feature.name)
                    )
                elif feature.dtype == FeatureType.datetime():
                    expr = (
                        pl.col(feature.name)
                        .dt.timestamp("ms")
                        .cast(pl.Utf8)
                        .alias(feature.name)
                    )
                elif feature.dtype.is_embedding or feature.dtype.is_array:
                    expr = pl.col(feature.name).map_elements(
                        lambda x: x.to_numpy().tobytes(), return_dtype=pl.Binary()
                    )

                request_data = request_data.with_columns(expr)
                features.append(feature.name)

            redis_frame = request_data.select(features).collect()

            for record in redis_frame.to_dicts():
                pipe.hset(
                    record["id"],
                    mapping={key: value for key, value in record.items() if value},
                )
                for key, value in record.items():
                    if value is None:
                        pipe.hdel(record["id"], key)
            await pipe.execute()

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        await self.insert(job, request)


@dataclass
class RedisStreamSource(StreamDataSource, SinkableDataSource, ColumnFeatureMappable):
    topic_name: str
    config: RedisConfig

    mapping_keys: dict[str, str] = field(default_factory=dict)
    record_coder: RecordCoder = field(default_factory=PassthroughRecordCoder)

    name: str = "redis"

    def with_coder(self, coder: RecordCoder) -> RedisStreamSource:
        self.record_coder = coder
        return self

    def consumer(self, from_timestamp: str | None = None) -> ReadableStream:
        from aligned.streams.redis import RedisStream

        return RedisStream(
            self.config.redis(),
            self.topic_name,
            record_coder=self.record_coder,
            read_timestamp=from_timestamp or "0-0",
        )

    def map_values(self, mappings: dict[str, str]) -> RedisStreamSource:
        self.mapping_keys = self.mapping_keys | mappings
        return self

    def make_redis_friendly(
        self, data: pl.LazyFrame, features: set[Feature]
    ) -> pl.LazyFrame:
        # Run one query per row
        for feature in features:
            expr = pl.col(feature.name)

            if feature.dtype == FeatureType.boolean():
                # Redis do not support bools
                expr = pl.col(feature.name).cast(pl.Int8, strict=False)
            elif feature.dtype == FeatureType.datetime():
                expr = pl.col(feature.name).dt.timestamp("ms")
            elif feature.dtype.is_embedding or feature.dtype.is_array:
                expr = pl.col(feature.name).map_elements(
                    lambda x: x.to_numpy().tobytes(), return_dtype=pl.Binary()
                )

            data = data.with_columns(expr.cast(pl.Utf8).alias(feature.name))

        return data

    async def write_to_stream(self, job: RetrievalJob) -> None:
        redis = self.config.redis()
        df = await job.to_lazy_polars()

        df = self.make_redis_friendly(
            df, job.request_result.features.union(job.request_result.entities)
        )
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
