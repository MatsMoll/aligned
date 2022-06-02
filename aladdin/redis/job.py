from dataclasses import dataclass

import pandas as pd

from aladdin.feature import Feature
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
                result = await redis.mget(keys)
                result_df.loc[mask, feature.name] = result

        return result_df

    async def to_df(self) -> pd.DataFrame:
        df = await self._to_df()
        df = await self.ensure_types(df)
        return await self.fill_missing(df)

    async def to_arrow(self) -> pd.DataFrame:
        return await super().to_arrow()
