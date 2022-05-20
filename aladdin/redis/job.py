from dataclasses import dataclass
from aladdin.redis.config import RedisConfig
from aladdin.retrival_job import FactualRetrivalJob
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.feature import Feature
import pandas as pd

def key(request: RetrivalRequest, entity: str, feature: Feature) -> str:
    return f"{request.feature_view_name}:{entity}:{feature.name}"

@dataclass
class FactualRedisJob(FactualRetrivalJob):

    config: RedisConfig
    requests: set[RetrivalRequest]
    facts: dict[str, list]

    async def _to_df(self) -> pd.DataFrame:
        redis = self.config.redis()

        columns = set()
        for request in self.requests:
            for feature in request.all_feature_names:
                columns.add(feature)

        result_df = pd.DataFrame(self.facts)
        
        for request in self.requests:
            entity_ids = result_df[request.entity_names]
            entities = [":".join(entity_ids) for _, entity_ids in entity_ids.iterrows()]
            for feature in request.all_features:
                # Fetch one column at a time
                keys = list()
                for entity in entities:
                    keys.append(key(request, entity, feature))
                result = await redis.mget(keys)
                result_df[feature.name] = result

        return result_df