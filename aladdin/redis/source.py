from dataclasses import dataclass
from aladdin.redis.config import RedisConfig
from aladdin.retrival_job import FactualRetrivalJob, RetrivalJob
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.feature import Feature
from aladdin.feature_source import FeatureSource
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

        sub_columns = list()
        
        for request in self.requests:
            entity_ids = result_df[request.entity_names]
            entities = [":".join(entity_ids) for _, entity_ids in entity_ids.iterrows()]
            for feature in request.all_features:
                # Fetch one column at a time
                keys = list()
                sub_columns.append(feature)
                for entity in entities:
                    keys.append(key(request, entity, feature))
                result = await redis.mget(keys)
                result_df[feature.name] = result

        return result_df

class RedisSource(FeatureSource):

    config: RedisConfig


    async def store(self, job: RetrivalJob, requests: set[RetrivalRequest]) -> None:

        redis = self.config.redis()
        data = await job.to_df()

        async with redis.pipeline(transaction=True) as pipe:
            for _, row in data.iterrows():
                # Run one query per row
                for request in requests:
                    entity_ids = row[request.entity_names]
                    entity_id = ":".join([str(entity_id) for entity_id in entity_ids])
                    if not entity_id:
                        continue

                    for feature in request.all_features:
                        value = row[feature.name]
                        if value and not pd.isnull(value):
                            if feature.dtype == bool:
                                # Redis do not support bools
                                value = int(value)
                            feature_key = key(request, entity_id, feature)
                            pipe.set(feature_key, value)

            await pipe.execute()


    def features_for(self, facts: dict[str, list], requests: set[RetrivalRequest]) -> RetrivalJob:
        return FactualRedisJob(self.config, requests, facts)