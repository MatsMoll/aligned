from dataclasses import dataclass

import numpy as np
import pandas as pd

from aligned.redis.config import RedisConfig
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import FactualRetrivalJob
from aligned.schemas.feature import FeatureType

try:
    import dask.dataframe as dd
except ModuleNotFoundError:
    import pandas as dd


@dataclass
class FactualRedisJob(FactualRetrivalJob):

    config: RedisConfig
    requests: list[RetrivalRequest]
    facts: dict[str, list]

    async def _to_df(self) -> pd.DataFrame:
        redis = self.config.redis()

        result_df = pd.DataFrame(self.facts)

        for request in self.requests:
            for entity in request.entities:
                if entity.dtype.is_numeric:
                    result_df[entity.name] = pd.to_numeric(result_df[entity.name], errors='coerce')
                else:
                    result_df[entity.name] = result_df[entity.name].convert_dtypes(
                        infer_objects=False, convert_string=True
                    )

        for request in self.requests:
            entity_ids = result_df[sorted(request.entity_names)]
            if entity_ids.empty:
                for feature in request.all_features:
                    result_df[feature.name] = np.nan
                continue

            mask = ~entity_ids.isna().any(axis=1)
            entities = (
                request.feature_view_name
                + ':'
                + entity_ids.loc[mask].astype('string').sum(axis=1).astype('string')
            )

            for feature in request.all_features:
                keys = entities + ':' + feature.name

                # If there is no entities, set feature to None
                if entities.empty:
                    result_df[feature.name] = np.nan
                    continue

                result = await redis.mget(keys.values)

                result_series = pd.Series(result)
                set_mask = mask.copy()
                result_value_mask = result_series.notna()
                set_mask[mask] = set_mask[mask] & (result_value_mask)

                if feature.dtype == FeatureType('').datetime:
                    dates = pd.to_datetime(result_series[result_value_mask], unit='s', utc=True)
                    result_df.loc[set_mask, feature.name] = dates
                elif feature.dtype == FeatureType('').bool:
                    result_df.loc[set_mask, feature.name] = result_series[result_value_mask] == '1'
                elif feature.dtype == FeatureType('').int32 or feature.dtype == FeatureType('').int64:
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
        return await self._to_df()

    async def _to_dask(self) -> dd.DataFrame:
        return await self._to_df()
