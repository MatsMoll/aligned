from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from pandas import DataFrame
from aladdin.data_source.batch_data_source import BatchDataSource

from aladdin.job_factory import JobFactory
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.repo_definition import RepoDefinition
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import CombineFactualJob, RetrivalJob
from aladdin.feature import Feature

@dataclass
class RawStringFeatureRequest:
    features: set[str]

    @property
    def feature_view_names(self) -> set[str]:
        return {RawStringFeatureRequest.unpack_feature(feature)[0] for feature in self.features}

    @property
    def grouped_features(self) -> dict[str, set[str]]:
        unpacked_features = [RawStringFeatureRequest.unpack_feature(feature) for feature in self.features]
        grouped = defaultdict(set)
        for feature_view, feature in unpacked_features:
            grouped[feature_view].add(feature)
        return grouped

    @staticmethod
    def unpack_feature(feature: str) -> tuple[str, str]:
        splits = feature.split(":")
        if len(splits) != 2:
            raise ValueError(f"Invalid feature name: {feature}")
        return (splits[0], splits[1])

@dataclass
class RedisConfig:

    url_env_var: str

    @property   
    def url(self) -> str:
        import os
        return os.environ[self.url_env_var]
    
    @staticmethod
    def localhost() -> "RedisConfig":
        import os
        os.environ["REDIS_URL"] = "redis://localhost:6379"
        return RedisConfig(url_env_var="REDIS_URL")

class RedisSource:

    config: RedisConfig

    async def store(self, job: RetrivalJob, requests: set[RetrivalRequest]) -> None:
        from redis.asyncio import Redis
        import pandas as pd

        redis = Redis.from_url(url=self.config.url)
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
                            feature_key = self.key(request, entity_id, feature)
                            pipe.set(feature_key, value)

                await pipe.execute()
            


    async def retrieve(self, facts: dict[str, list], requests: list[RetrivalRequest]) -> DataFrame:
        from redis.asyncio import Redis
        redis = Redis.from_url(url=self.config.url)

        columns = set()
        for request in requests:
            for feature in request.all_feature_names:
                columns.add(feature)

        result_df = DataFrame(facts)

        sub_columns = list()
        
        for request in requests:
            entity_ids = result_df[request.entity_names]
            entities = [":".join(entity_ids) for _, entity_ids in entity_ids.iterrows()]
            for feature in request.all_features:
                # Fetch one column at a time
                keys = list()
                sub_columns.append(feature)
                for entity in entities:
                    keys.append(self.key(request, entity, feature))
                result = await redis.mget(keys)
                result_df[feature.name] = result

        return result_df

    def key(self, request: RetrivalRequest, entity: str, feature: Feature) -> str:
        return f"{request.feature_view_name}:{entity}:{feature.name}"

@dataclass
class OnlineStore:
    
    storage: RedisSource
    feature_views: dict[str, CompiledFeatureView]

    async def store(self, job: RetrivalJob, requests: set[RetrivalRequest]) -> None:
        pass

    async def features_for(self, facts: dict[str, list], features: list[str]) -> DataFrame:
        feature_request = RawStringFeatureRequest(features=set(features))
        requests = [self.feature_views[view].request_for(features) for view, features in feature_request.grouped_features.items()]
        return await self.storage.retrieve(facts, requests)


class FeatureStore:

    job_factories: dict[str, JobFactory]
    feature_views: dict[str, CompiledFeatureView]
    model_requests: dict[str, dict[BatchDataSource, RetrivalRequest]]

    @property
    def all_models(self) -> list[str]:
        return list(self.model_requests.keys())


    def __init__(self, feature_views: set[CompiledFeatureView], models: dict[str, list[str]], job_factories: set[JobFactory]) -> None:
        self.job_factories = {factory.source.type_name: factory for factory in job_factories}
        self.feature_views = {fv.name: fv for fv in feature_views}
        self.model_requests = {name: self.requests_for(model) for name, model in models.items()}


    @staticmethod
    def from_definition(repo: RepoDefinition, job_factories: set[JobFactory] | None = None) -> "FeatureStore":
        
        import job_factories as jf
        if job_factories:
            jf.custom_factories.update(job_factories)

        return FeatureStore(
            jf.get_factories(),
            feature_views=repo.feature_views,
            models=repo.models
        )


    def features_for(self, facts: dict[str, list], features: list[str]) -> RetrivalJob:
        requests = self.requests_for(features)
        return self.job_for(facts, requests)

    def model(self, name: str) -> "OfflineModelStore":
        return OfflineModelStore(
            self.job_factories,
            self.model_requests[name]
        )


    def requests_for(self, features: list[str]) -> dict[BatchDataSource, RetrivalRequest]:
        feature_request = RawStringFeatureRequest(features=set(features))
        feature_views: set[CompiledFeatureView] = set()
        features = feature_request.grouped_features
        requests: dict[BatchDataSource, RetrivalRequest] = {}
        for feature_view_name in feature_request.feature_view_names:
            feature_view = self.feature_views[feature_view_name]
            feature_views.add(feature_view)
            requests[feature_view.batch_data_source] = feature_view.request_for(features[feature_view_name])
        return requests

    
    def job_for(self, facts: dict[str, list], requests: dict[BatchDataSource, RetrivalRequest]) -> RetrivalJob:
        job_factories = {self.job_factories[source.type_name] for source in requests.keys()}
        return CombineFactualJob([factory.facts(facts=facts, sources=requests) for factory in job_factories])

@dataclass
class OfflineModelStore:

    job_factories: dict[str, JobFactory]
    requests: dict[BatchDataSource, RetrivalRequest]

    def features_for(self, facts: dict[str, list]) -> RetrivalJob:
        job_factories = {self.job_factories[source.type_name] for source in self.requests.keys()}
        return CombineFactualJob([factory.facts(facts=facts, sources=self.requests) for factory in job_factories])

    async def write(self, values: dict[str]):
        pass

@dataclass
class OnlineModelStore:

    storage: RedisSource
    requests: list[RetrivalRequest]

    async def features_for(self, entities: dict[str, list]) -> DataFrame:
        return await self.storage.retrieve(entities, self.requests)

    async def write(self, values: dict[str]):
        pass