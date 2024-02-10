from dataclasses import dataclass

import pandas as pd
import polars as pl

from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import RequestResult, RetrivalJob
from aligned.schemas.feature import FeatureType
from aligned.sources.redis import RedisConfig


@dataclass
class FactualRedisJob(RetrivalJob):

    config: RedisConfig
    requests: list[RetrivalRequest]
    facts: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.requests

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    def describe(self) -> str:
        features_to_load = [list(request.all_feature_names) for request in self.requests]
        return f'Loading features from Redis using HMGET {features_to_load}'

    async def to_lazy_polars(self) -> pl.LazyFrame:
        redis = self.config.redis()

        result_df = (await self.facts.to_lazy_polars()).collect()

        for request in self.requests:
            redis_combine_id = 'redis_combine_entity_id'
            entities = result_df.select(
                [
                    (
                        pl.concat_str(
                            [pl.lit(request.location.identifier), pl.lit(':')]
                            + [pl.col(col) for col in sorted(request.entity_names)]
                        )
                    ).alias(redis_combine_id),
                    pl.col(list(request.entity_names)),
                ]
            ).filter(pl.col(redis_combine_id).is_not_null())

            if entities.shape[0] == 0:
                # Do not connect to redis if there are no entities to fetch
                result_df = result_df.with_columns(
                    [pl.lit(None).alias(column.name) for column in request.all_features]
                )
                continue

            features = list({feature.name for feature in request.returned_features})

            async with redis.pipeline(transaction=False) as pipe:
                for entity in entities[redis_combine_id]:
                    pipe.hmget(entity, keys=features)
                result = await pipe.execute()

            reqs: pl.DataFrame = pl.concat(
                [entities, pl.DataFrame(result, schema=features, orient='row')], how='horizontal'
            ).select(pl.exclude(redis_combine_id))

            for feature in request.returned_features:
                if feature.dtype == FeatureType.bool():
                    reqs = reqs.with_columns(pl.col(feature.name).cast(pl.Int8).cast(pl.Boolean))
                elif reqs[feature.name].dtype == pl.Utf8 and (
                    feature.dtype == FeatureType.int32() or feature.dtype == FeatureType.int64()
                ):
                    reqs = reqs.with_columns(
                        pl.col(feature.name)
                        .str.splitn('.', 2)
                        .struct.field('field_0')
                        .cast(feature.dtype.polars_type)
                        .alias(feature.name)
                    )
                elif feature.dtype == FeatureType.embedding() or feature.dtype == FeatureType.array():
                    import numpy as np

                    reqs = reqs.with_columns(pl.col(feature.name).apply(lambda row: np.frombuffer(row)))
                else:
                    reqs = reqs.with_columns(pl.col(feature.name).cast(feature.dtype.polars_type))
                # if feature.dtype == FeatureType.datetime():
                #     dates = pd.to_datetime(result_series[result_value_mask], unit='s', utc=True)
                #     result_df.loc[set_mask, feature.name] = dates
                # elif feature.dtype == FeatureType.embedding():
                #     result_df.loc[set_mask, feature.name] = (
                #         result_series[result_value_mask].str.split(',')
                # .apply(lambda x: [float(i) for i in x])
                #     )

            result_df = result_df.join(reqs, on=list(request.entity_names), how='left')

        return result_df.lazy()
