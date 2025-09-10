from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from aligned.lazy_imports import pandas as pd
from aligned.request.retrieval_request import RetrievalRequest
from aligned.retrieval_job import RequestResult, RetrievalJob
from aligned.schemas.date_formatter import DateFormatter
from aligned.schemas.feature import FeatureType
from aligned.sources.redis import RedisConfig


@dataclass
class FactualRedisJob(RetrievalJob):
    config: RedisConfig
    requests: list[RetrievalRequest]
    facts: RetrievalJob
    formatter: DateFormatter

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    @property
    def retrieval_requests(self) -> list[RetrievalRequest]:
        return self.requests

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    def describe(self) -> str:
        features_to_load = [
            list(request.all_feature_names) for request in self.requests
        ]
        return f"Loading features from Redis using HMGET {features_to_load}"

    async def to_lazy_polars(self) -> pl.LazyFrame:
        redis = self.config.redis()

        result_df = (await self.facts.to_lazy_polars()).collect()

        for request in self.requests:
            redis_combine_id = "redis_combine_entity_id"

            needed_features = request.all_required_features

            entities = result_df.select(
                [
                    (
                        pl.concat_str(
                            [pl.lit(request.location.identifier), pl.lit(":")]
                            + [pl.col(col) for col in sorted(request.entity_names)]
                        )
                    ).alias(redis_combine_id),
                    pl.col(list(request.entity_names)),
                ]
            ).filter(pl.col(redis_combine_id).is_not_null())

            if entities.shape[0] == 0:
                # Do not connect to redis if there are no entities to fetch
                result_df = result_df.with_columns(
                    [pl.lit(None).alias(column.name) for column in needed_features]
                )
                continue

            features = list(feature.name for feature in needed_features)

            async with redis.pipeline(transaction=False) as pipe:
                for entity in entities[redis_combine_id]:
                    pipe.hmget(entity, keys=features)
                result = await pipe.execute()

            reqs: pl.DataFrame = pl.concat(
                [entities, pl.DataFrame(result, schema=features, orient="row")],
                how="horizontal",
            ).select(pl.exclude(redis_combine_id))

            org_schema = reqs.collect_schema()

            for feature in needed_features:
                if org_schema[feature.name] == pl.Null():
                    reqs = reqs.with_columns(
                        pl.col(feature.name).cast(feature.dtype.polars_type)
                    )
                    continue

                if feature.dtype.is_datetime:
                    reqs = reqs.with_columns(self.formatter.decode_polars(feature.name))
                elif feature.dtype == FeatureType.boolean():
                    reqs = reqs.with_columns(
                        pl.col(feature.name).cast(pl.Int8).cast(pl.Boolean)
                    )
                elif org_schema[feature.name] == pl.Utf8 and (
                    feature.dtype == FeatureType.int32()
                    or feature.dtype == FeatureType.int64()
                ):
                    reqs = reqs.with_columns(
                        pl.col(feature.name)
                        .str.splitn(".", 2)
                        .struct.field("field_0")
                        .cast(feature.dtype.polars_type)
                        .alias(feature.name)
                    )
                elif feature.dtype.is_embedding:
                    import numpy as np

                    reqs = reqs.with_columns(
                        pl.col(feature.name).map_elements(
                            lambda row: np.frombuffer(row)
                        )
                    )
                elif feature.dtype.is_array:
                    reqs = reqs.with_columns(
                        pl.col(feature.name).str.json_decode(feature.dtype.polars_type)
                    )
                else:
                    reqs = reqs.with_columns(
                        pl.col(feature.name).cast(feature.dtype.polars_type)
                    )

            result_df = result_df.join(reqs, on=list(request.entity_names), how="left")

        return result_df.lazy()
