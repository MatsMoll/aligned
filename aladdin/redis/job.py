from dataclasses import dataclass

import numpy as np
import pandas as pd

from aladdin.feature import Feature, FeatureType
from aladdin.redis.config import RedisConfig
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import FactualRetrivalJob


def key(request: RetrivalRequest, entity: str, feature: Feature) -> str:
    return f'{request.feature_view_name}:{entity}:{feature.name}'


@dataclass
class FactualRedisJob(FactualRetrivalJob):

    config: RedisConfig
    requests: list[RetrivalRequest]
    facts: dict[str, list]

    async def _to_df(self) -> pd.DataFrame:
        redis = self.config.redis()

        columns = set()
        for request in self.requests:
            for feature in request.all_feature_names:
                columns.add(feature)

        result_df = pd.DataFrame(self.facts)

        for request in self.requests:
            entity_ids = result_df[list(request.entity_names)]
            mask = ~entity_ids.isna().any(axis=1)
            entities = [':'.join(entity_ids) for _, entity_ids in entity_ids.loc[mask].iterrows()]
            for feature in request.all_features:
                # Fetch one column at a time
                keys = []
                for entity in entities:
                    keys.append(key(request, entity, feature))

                # If there is no entities, set feature to None
                if not entities:
                    result_df[feature.name] = np.nan
                    continue

                result = await redis.mget(keys)
                result_series = pd.Series(result)
                set_mask = mask.copy()
                result_value_mask = result_series.notna()
                set_mask[mask] = set_mask[mask] & (result_value_mask)

                if feature.dtype == FeatureType('').datetime:
                    dates = pd.to_datetime(result_series[result_value_mask], unit='s', utc=True)
                    result_df.loc[set_mask, feature.name] = dates
                elif feature.dtype == FeatureType('').int32 or FeatureType('').int64:
                    result_df.loc[set_mask, feature.name] = (
                        result_series[result_value_mask]
                        .str.split('.', n=1)
                        .str[0]
                        .astype(feature.dtype.pandas_type)
                    )
                else:
                    result_df.loc[set_mask, feature.name] = result_series[result_value_mask].astype(
                        feature.dtype.pandas_type
                    )

        return result_df

    async def to_df(self) -> pd.DataFrame:
        df = await self._to_df()
        return await self.fill_missing(df)

    async def to_arrow(self) -> pd.DataFrame:
        return await super().to_arrow()
