import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from aladdin.feature_source import (
    BatchFeatureSource,
    FeatureSource,
    RangeFeatureSource,
    WritableFeatureSource,
)
from aladdin.feature_view.combined_view import CombinedFeatureView, CompiledCombinedFeatureView
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.feature_view.feature_view import FeatureView
from aladdin.model import ModelService
from aladdin.online_source import BatchOnlineSource
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
    event_timestamp_column = 'event_timestamp'

    @property
    def all_models(self) -> list[str]:
        return list(self.model_requests.keys())

    def __init__(
        self,
        feature_views: dict[str, CompiledFeatureView],
        combined_feature_views: dict[str, CompiledCombinedFeatureView],
        models: dict[str, FeatureRequest],
        feature_source: FeatureSource,
    ) -> None:
        self.feature_source = feature_source
        self.combined_feature_views = combined_feature_views
        self.feature_views = feature_views
        self.model_requests = models

    @staticmethod
    def experimental() -> 'FeatureStore':
        return FeatureStore.from_definition(RepoDefinition(set(), set(), {}, BatchOnlineSource()))

    @staticmethod
    def from_definition(repo: RepoDefinition, feature_source: FeatureSource | None = None) -> 'FeatureStore':
        source = feature_source or repo.online_source.feature_source(repo.feature_views)
        feature_views = {fv.name: fv for fv in repo.feature_views}
        combined_feature_views = {fv.name: fv for fv in repo.combined_feature_views}
        return FeatureStore(
            feature_views=feature_views,
            combined_feature_views=combined_feature_views,
            models={
                name: FeatureStore._requests_for(
                    RawStringFeatureRequest(model), feature_views, combined_feature_views
                )
                for name, model in repo.models.items()
            },
            feature_source=source,
        )

    @staticmethod
    async def from_reference_at_path(path: str = '.') -> 'FeatureStore':
        """Looks for a file reference struct, and loads the associated repo.

        This can be used for changing which feature store definitions
        to read based on defined enviroment variables.

        If you rather want to generate a feature store based on a dir,
        then consider using `FeatureStore.from_dir(...)` instead.

        Args:
            path (str, optional): The path of the dir to search. Defaults to ".".

        Returns:
            FeatureStore: A feature store based on the feature references
        """
        repo_def = await RepoDefinition.from_reference_at_path(path)
        return FeatureStore.from_definition(repo_def)

    @staticmethod
    async def from_dir(path: str = '.') -> 'FeatureStore':
        """Reads and generates a feature store based on the given directory's content.

        This will read the feature views, services etc in a given repo and generate a feature store.
        This can be used for fast development purposes.

        If you rather want a more flexible deployable solution.
        Consider using `FeatureStore.from_reference_at_path(...)` which will can read an existing
        generated file from differnet storages, based on an enviroment variable.

        Args:
            path (str, optional): the directory to read from. Defaults to ".".

        Returns:
            FeatureStore: The generated feature store
        """
        return FeatureStore.from_definition(RepoDefinition.from_path(path))

    def features_for(self, entities: dict[str, list], features: list[str]) -> RetrivalJob:

        if self.event_timestamp_column not in entities:
            raise ValueError(f'Missing {self.event_timestamp_column} in entities')

        feature_request = RawStringFeatureRequest(features=set(features))
        entities_names: set[str] = set(entities.keys())
        requests = self.requests_for(feature_request)
        feature_names = set()
        for feature_set in feature_request.grouped_features.values():
            feature_names.update(feature_set.union({self.event_timestamp_column}))
        for request in requests.needed_requests:
            feature_names.update(request.entity_names)
            entities_names.update(request.entity_names)

        return FilterJob(
            feature_request.feature_names.union(entities_names),
            self.feature_source.features_for(entities, requests),
        )

    def model(self, name: str) -> 'ModelFeatureStore':
        request = self.model_requests[name]
        return ModelFeatureStore(self.feature_source, request)

    @staticmethod
    def _requests_for(
        feature_request: RawStringFeatureRequest,
        feature_views: dict[str, CompiledFeatureView],
        combined_feature_views: dict[str, CompiledCombinedFeatureView],
    ) -> FeatureRequest:
        features = feature_request.grouped_features
        requests: list[RetrivalRequest] = []
        entity_names = set()
        for feature_view_name in feature_request.feature_view_names:
            if feature_view_name in combined_feature_views:
                cfv = combined_feature_views[feature_view_name]
                requests.extend(cfv.requests_for(features[feature_view_name]).needed_requests)
            else:
                feature_view = feature_views[feature_view_name]
                sub_requests = feature_view.request_for(features[feature_view_name]).needed_requests
                requests.extend(sub_requests)
                for request in sub_requests:
                    entity_names.update(request.entity_names)

        if len(requests) > 1:
            entity_names.add('event_timestamp')

        return FeatureRequest(
            'some_name',
            feature_request.feature_names.union(entity_names),
            RetrivalRequest.combine(requests),
        )

    def requests_for(self, feature_request: RawStringFeatureRequest) -> FeatureRequest:
        return FeatureStore._requests_for(feature_request, self.feature_views, self.combined_feature_views)

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

    def offline_store(self) -> 'FeatureStore':
        return FeatureStore(
            feature_views=self.feature_views,
            combined_feature_views=self.combined_feature_views,
            models=self.model_requests,
            feature_source=BatchOnlineSource().feature_source(set(self.feature_views.values())),
        )


@dataclass
class ModelFeatureStore:

    source: FeatureSource
    request: FeatureRequest

    def features_for(self, entities: dict[str, list]) -> RetrivalJob:

        if 'event_timestamp' not in entities:
            raise ValueError('Missing event_timestamp in entities')

        return FilterJob(
            self.request.features_to_include,
            self.source.features_for(entities, self.request),
        )

    async def write(self, values: dict[str, Any]) -> None:
        raise NotImplementedError()


@dataclass
class FeatureViewStore:

    source: FeatureSource
    view: CompiledFeatureView

    def all(self, limit: int | None = None) -> RetrivalJob:
        if not isinstance(self.source, RangeFeatureSource):
            raise ValueError('The source needs to conform to RangeFeatureSource')
        return self.source.all_for(self.view.request_all, limit)

    def between(self, start_date: datetime, end_date: datetime) -> RetrivalJob:
        if not isinstance(self.source, RangeFeatureSource):
            raise ValueError('The source needs to conform to RangeFeatureSource')
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

        # As it is a feature view should it only contain one request
        request = self.view.request_all.needed_requests[0]
        df = DataFrame(values)

        if request.entity_names - set(df.columns):
            missing = request.entity_names - set(df.columns)
            raise ValueError(f'Missing entities: {missing}')

        if request.all_required_feature_names - set(df.columns):
            missing = request.all_required_feature_names - set(df.columns)
            logger.info(
                'Some features is missing.',
                f'Will fill values with None, but it could be a potential problem: {missing}',
            )
            df[list(missing)] = None

        await self.source.write(FileFullJob(LiteralReference(df), request, limit=None), [request])
