from __future__ import annotations

from datetime import timedelta
import logging
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import polars as pl

from aligned.config_value import ConfigValue, EnvironmentValue, LiteralValue
from aligned.schemas.date_formatter import DateFormatter
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.schemas.transformation import Expression
from aligned.streams.interface import ReadableStream
from aligned.lazy_imports import redis
from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    ColumnFeatureMappable,
    AsBatchSource,
)
from aligned.data_source.stream_data_source import SinkableDataSource, StreamDataSource
from aligned.feature_source import WritableFeatureSource
from aligned.request.retrieval_request import RetrievalRequest
from aligned.retrieval_job import RetrievalJob
from aligned.schemas.codable import Codable
from aligned.schemas.feature import Feature, FeatureType
from aligned.schemas.record_coders import PassthroughRecordCoder, RecordCoder
from aligned.schemas.vector_storage import VectorIndex, VectorStorage

if TYPE_CHECKING:
    from redis.client import AbstractRedis
    from redis.commands.search.field import (
        NumericField,
        TagField,
        TextField,
        VectorField,
    )


logger = logging.getLogger(__name__)


redis_manager: dict[str, AbstractRedis] = {}


@dataclass
class RedisConfig(Codable, AsBatchSource):
    redis_url: ConfigValue

    @property
    def url(self) -> str:
        return self.redis_url.read()

    @staticmethod
    def from_url(url: str) -> RedisConfig:
        return RedisConfig(LiteralValue(url))

    @staticmethod
    def from_env(
        environment_key: str = "REDIS_URL",
        default_url: str | None = "redis://localhost:6379",
    ) -> RedisConfig:
        return RedisConfig(EnvironmentValue(environment_key, default_value=default_url))

    @staticmethod
    def localhost() -> RedisConfig:
        return RedisConfig.from_env()

    def as_source(self) -> CodableBatchDataSource:
        return RedisSource(self)

    def redis(self) -> AbstractRedis:
        url = self.url
        if url not in redis_manager:
            from redis import Redis as SyncRedis

            sync_client = SyncRedis.from_url(url, decode_responses=True)
            info = sync_client.info()
            if "cluster_enabled" in info and bool(info["cluster_enabled"]):
                redis_manager[url] = redis.RedisCluster.from_url(
                    url=url, decode_responses=True
                )
            else:
                redis_manager[url] = redis.StrictRedis.from_url(
                    url, decode_responses=True
                )

        return redis_manager[url]

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


ValkeyConfig = RedisConfig


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
class RedisSource(WritableFeatureSource, CodableBatchDataSource):
    config: RedisConfig
    batch_size = 1_000_000
    expire_duration: timedelta | None = None
    formatter: DateFormatter = field(default_factory=DateFormatter.iso_8601)

    type_name: str = "redis_source"

    def job_group_key(self) -> str:
        """
        A key defining which sources can be grouped together in one request.
        """
        return self.type_name

    def needed_configs(self) -> list[ConfigValue]:
        return [self.config.redis_url]

    def with_view(self, view: CompiledFeatureView) -> RedisSource:
        if self.expire_duration is None and view.unacceptable_freshness:
            self.expire_duration = view.unacceptable_freshness
        return self

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls: type[RedisSource],
        facts: RetrievalJob,
        requests: list[tuple[RedisSource, RetrievalRequest]],
    ) -> RetrievalJob:
        from aligned.redis.job import FactualRedisJob

        source = requests[0][0]
        requested_features = [req for (_, req) in requests]

        return FactualRedisJob(
            source.config, requested_features, facts, formatter=source.formatter
        )

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        from aligned.redis.job import FactualRedisJob

        return FactualRedisJob(self.config, [request], facts, formatter=self.formatter)

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        redis = self.config.redis()
        data = await job.to_lazy_polars()

        async with redis.pipeline(transaction=False) as pipe:
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

            def encode_list(value: pl.Series) -> str:
                return json.dumps(value.to_list())

            for feature in request.all_required_features:
                expr = pl.col(feature.name).cast(pl.Utf8).alias(feature.name)

                if feature.dtype == FeatureType.boolean():
                    # Redis do not support bools
                    expr = (
                        pl.col(feature.name)
                        .cast(pl.Int8, strict=False)
                        .cast(pl.Utf8)
                        .alias(feature.name)
                    )
                elif feature.dtype.is_datetime or (
                    feature.dtype == FeatureType.datetime()
                ):
                    expr = self.formatter.encode_polars(feature.name).alias(
                        feature.name
                    )
                elif feature.dtype.is_embedding:
                    expr = pl.col(feature.name).map_elements(
                        lambda x: x.to_numpy().tobytes(), return_dtype=pl.Binary()
                    )
                elif feature.dtype.is_array:
                    expr = pl.col(feature.name).map_elements(
                        encode_list, return_dtype=pl.String()
                    )

                request_data = request_data.with_columns(expr)
                features.append(feature.name)

            redis_frame = request_data.select(features).collect()

            for record in redis_frame.to_dicts():
                record_key = record["id"]
                pipe.hset(
                    record_key,
                    mapping={key: value for key, value in record.items() if value},
                )
                for key, value in record.items():
                    if value is None:
                        pipe.hdel(record_key, key)

                if self.expire_duration:
                    pipe.expire(record_key, time=self.expire_duration)

            await pipe.execute()

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        await self.insert(job, request)

    async def overwrite(
        self,
        job: RetrievalJob,
        request: RetrievalRequest,
        predicate: Expression | None = None,
    ) -> None:
        return await self.insert(job, request)


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
