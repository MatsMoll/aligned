from dataclasses import dataclass

import pandas as pd
import polars as pl

from aligned.redis.config import RedisConfig
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import FactualRetrivalJob, RequestResult
from aligned.schemas.feature import FeatureType


@dataclass
class FactualRedisJob(FactualRetrivalJob):

    config: RedisConfig
    requests: list[RetrivalRequest]
    facts: dict[str, list]

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_polars()).collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        redis = self.config.redis()

        result_df = pl.DataFrame(self.facts)

        for request in self.requests:
            entities = result_df.select(
                (
                    pl.lit(request.feature_view_name)
                    + pl.lit(':')
                    + pl.concat_str(sorted(request.entity_names))
                ).alias('id')
            )

            features = request.all_feature_names

            async with redis.pipeline(transaction=False) as pipe:
                for entity in entities['id']:
                    pipe.hmget(entity, keys=features)
                result = await pipe.execute()

            reqs = pl.DataFrame(result, columns=features)
            for feature in request.all_features:
                if feature.dtype == FeatureType('').bool:
                    reqs = reqs.with_column(pl.col(feature.name).cast(pl.Int8).cast(pl.Boolean))
                elif feature.dtype == FeatureType('').int32 or feature.dtype == FeatureType('').int64:
                    reqs = reqs.with_column(
                        pl.col(feature.name)
                        .str.splitn('.', 2)
                        .struct.field('field_0')
                        .cast(feature.dtype.polars_type)
                        .alias(feature.name)
                    )
                else:
                    reqs = reqs.with_column(pl.col(feature.name).cast(feature.dtype.polars_type))
                # if feature.dtype == FeatureType('').datetime:
                #     dates = pd.to_datetime(result_series[result_value_mask], unit='s', utc=True)
                #     result_df.loc[set_mask, feature.name] = dates
                # elif feature.dtype == FeatureType('').embedding:
                #     result_df.loc[set_mask, feature.name] = (
                #         result_series[result_value_mask].str.split(',')
                # .apply(lambda x: [float(i) for i in x])
                #     )

            result_df = pl.concat([reqs, result_df], how='horizontal')

        return result_df.lazy()
