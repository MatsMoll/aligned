import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from aladdin.feature_source import BatchFeatureSource, FeatureSource, WritableFeatureSource
from aladdin.feature_view.combined_view import CombinedFeatureView, CompiledCombinedFeatureView
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.feature_view.feature_view import FeatureView
from aladdin.model import ModelService
from aladdin.repo_definition import RepoDefinition
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest
from aladdin.retrival_job import FilterJob, RetrivalJob

logger = logging.getLogger(__name__)


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

    @property
    def feature_names(self) -> set[str]:
        return {RawStringFeatureRequest.unpack_feature(feature)[1] for feature in self.features}

    @staticmethod
    def unpack_feature(feature: str) -> tuple[str, str]:
        splits = feature.split(':')
        if len(splits) != 2:
            raise ValueError(f'Invalid feature name: {feature}')
        return (splits[0], splits[1])


class FeatureStore:

    feature_source: FeatureSource
    feature_views: dict[str, CompiledFeatureView]
    combined_feature_views: dict[str, CompiledCombinedFeatureView]
    model_requests: dict[str, FeatureRequest]

    @property
    def all_models(self) -> list[str]:
        return list(self.model_requests.keys())

    def __init__(
        self,
        feature_views: set[CompiledFeatureView],
        combined_feature_views: set[CompiledCombinedFeatureView],
        models: dict[str, set[str]],
        feature_source: FeatureSource,
    ) -> None:
        self.feature_source = feature_source
        self.combined_feature_views = {fv.name: fv for fv in combined_feature_views}
        self.feature_views = {fv.name: fv for fv in feature_views}
        self.model_requests = {
            name: self.requests_for(RawStringFeatureRequest(model)) for name, model in models.items()
        }

    @staticmethod
    def from_definition(repo: RepoDefinition, feature_source: FeatureSource | None = None) -> 'FeatureStore':
        source = feature_source or repo.online_source.feature_source(repo.feature_views)
        return FeatureStore(
            feature_views=repo.feature_views,
            combined_feature_views=repo.combined_feature_views,
            models=repo.models,
            feature_source=source,
        )

    @staticmethod
    async def from_dir(path: str) -> 'FeatureStore':
        repo_def = await RepoDefinition.from_path(path)
        return FeatureStore.from_definition(repo_def)

    def features_for(self, facts: dict[str, list], features: list[str]) -> RetrivalJob:
        feature_request = RawStringFeatureRequest(features=set(features))
        entities = set()
        requests = self.requests_for(feature_request)
        feature_names = set()
        for feature_set in feature_request.grouped_features.values():
            feature_names.update(feature_set)
        for request in requests.needed_requests:
            feature_names.update(request.entity_names)
            entities.update(request.entity_names)

        return FilterJob(
            feature_request.feature_names.union(entities),
            self.feature_source.features_for(facts, requests),
        )

    def model(self, name: str) -> 'OfflineModelStore':
        request = self.model_requests[name]
        return OfflineModelStore(self.feature_source, request)

    def requests_for(self, feature_request: RawStringFeatureRequest) -> FeatureRequest:
        features = feature_request.grouped_features
        requests: list[RetrivalRequest] = []
        for feature_view_name in feature_request.feature_view_names:
            if feature_view_name in self.combined_feature_views:
                cfv = self.combined_feature_views[feature_view_name]
                requests.extend(cfv.requests_for(features[feature_view_name]).needed_requests)
            else:
                feature_view = self.feature_views[feature_view_name]
                requests.extend(feature_view.request_for(features[feature_view_name]).needed_requests)
        return FeatureRequest(
            'some_name',
            feature_request.feature_names,
            RetrivalRequest.combine(requests),
        )

    def feature_view(self, view: str) -> 'FeatureViewStore':
        view = self.feature_views[view]
        return FeatureViewStore(self.feature_source, view)

    def add_feature_view(self, feature_view: FeatureView) -> None:
        compiled_view = type(feature_view).compile()
        self.feature_views[compiled_view.name] = compiled_view
        if isinstance(self.feature_source, BatchFeatureSource):
            self.feature_source.sources[compiled_view.name] = compiled_view.batch_data_source

    def add_combined_feature_view(self, feature_view: CombinedFeatureView) -> None:
        compiled_view = type(feature_view).compile()
        self.combined_feature_views[compiled_view.name] = compiled_view

    def add_model_service(self, service: ModelService) -> None:
        request = RawStringFeatureRequest(service.feature_refs)
        self.model_requests[service.name] = self.requests_for(request)

    def all_for(self, view: str, limit: int | None = None) -> RetrivalJob:
        return self.feature_source.all_for(self.feature_views[view].request_all, limit)


@dataclass
class OfflineModelStore:

    source: FeatureSource
    request: FeatureRequest

    def features_for(self, facts: dict[str, list]) -> RetrivalJob:

        return FilterJob(
            self.request.features_to_include,
            self.source.features_for(facts, self.request),
        )

    async def write(self, values: dict[str, Any]) -> None:
        pass


@dataclass
class FeatureViewStore:

    source: FeatureSource
    view: CompiledFeatureView

    def all(self, limit: int | None = None) -> RetrivalJob:
        return self.source.all_for(self.view.request_all, limit)

    def between(self, start_date: datetime, end_date: datetime) -> RetrivalJob:
        return self.source.all_between(start_date, end_date, self.view.request_all)

    def previous(self, days: int = 0, minutes: int = 0, seconds: int = 0) -> RetrivalJob:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days, minutes=minutes, seconds=seconds)
        return self.between(start_date, end_date)

    async def write(self, values: dict[str, Any]) -> None:
        if not isinstance(self.source, WritableFeatureSource):
            logger.info('Feature Source is not writable')
            return

        from pandas import DataFrame

        from aladdin.local.job import FileFullJob
        from aladdin.local.source import LiteralReference

        request = self.view.request_all.needed_requests[0]
        df = DataFrame(values)

        if request.all_required_feature_names - set(df.columns):
            missing = request.all_required_feature_names - set(df.columns)
            raise ValueError(
                f'Missing some required features: {request.all_required_feature_names},'
                f' missing: {missing}'
            )

        await self.source.write(FileFullJob(LiteralReference(df), request, limit=None), [request])
