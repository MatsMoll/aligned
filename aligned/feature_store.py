import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import import_module
from typing import Any

from aligned.compiler.model import Model
from aligned.data_file import DataFileReference
from aligned.enricher import Enricher
from aligned.exceptions import CombinedFeatureViewQuerying
from aligned.feature_source import (
    BatchFeatureSource,
    FeatureSource,
    RangeFeatureSource,
    WritableFeatureSource,
)
from aligned.feature_view.combined_view import CombinedFeatureView, CompiledCombinedFeatureView
from aligned.feature_view.feature_view import FeatureView
from aligned.online_source import BatchOnlineSource, OnlineSource
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.retrival_job import FilterJob, RetrivalJob, StreamAggregationJob, SupervisedJob
from aligned.schemas.feature import FeatureLocation
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.schemas.model import EventTrigger
from aligned.schemas.model import Model as ModelSchema
from aligned.schemas.repo_definition import EnricherReference, RepoDefinition, RepoMetadata

logger = logging.getLogger(__name__)


@dataclass
class RawStringFeatureRequest:
    features: set[str]

    @property
    def locations(self) -> set[FeatureLocation]:
        return {RawStringFeatureRequest.unpack_feature(feature)[0] for feature in self.features}

    @property
    def grouped_features(self) -> dict[FeatureLocation, set[str]]:
        unpacked_features = [RawStringFeatureRequest.unpack_feature(feature) for feature in self.features]
        grouped = defaultdict(set)
        for feature_view, feature in unpacked_features:
            grouped[feature_view].add(feature)
        return grouped

    @property
    def feature_names(self) -> set[str]:
        return {RawStringFeatureRequest.unpack_feature(feature)[1] for feature in self.features}

    @staticmethod
    def unpack_feature(feature: str) -> tuple[FeatureLocation, str]:
        splits = feature.split(':')
        if len(splits) == 3:
            return (FeatureLocation(splits[1], splits[0]), splits[2])
        if len(splits) == 2:
            return (FeatureLocation(splits[0], 'feature_view'), splits[1])
        else:
            raise ValueError(f'Unable to decode {splits}')


class FeatureStore:

    feature_source: FeatureSource
    feature_views: dict[str, CompiledFeatureView]
    combined_feature_views: dict[str, CompiledCombinedFeatureView]
    models: dict[str, ModelSchema]
    event_timestamp_column = 'event_timestamp'

    @property
    def all_models(self) -> list[str]:
        return list(self.models.keys())

    def __init__(
        self,
        feature_views: dict[str, CompiledFeatureView],
        combined_feature_views: dict[str, CompiledCombinedFeatureView],
        models: dict[str, ModelSchema],
        feature_source: FeatureSource,
    ) -> None:
        self.feature_source = feature_source
        self.combined_feature_views = combined_feature_views
        self.feature_views = feature_views
        self.models = models

    @staticmethod
    def experimental() -> 'FeatureStore':
        from aligned.online_source import BatchOnlineSource

        return FeatureStore.from_definition(
            RepoDefinition(
                metadata=RepoMetadata(created_at=datetime.utcnow(), name='experimental'),
                online_source=BatchOnlineSource(),
            )
        )

    @staticmethod
    def register_enrichers(enrichers: list[EnricherReference]) -> None:
        from types import ModuleType

        class DynamicEnricher(ModuleType):
            def __init__(self, values: dict[str, Enricher]) -> None:
                for key, item in values.items():
                    self.__setattr__(key, item)

        def set_module(path: str, module_class: DynamicEnricher) -> None:
            import sys

            components = path.split('.')
            cum_path = ''

            for component in components:
                cum_path += f'.{component}'
                if cum_path.startswith('.'):
                    cum_path = cum_path[1:]

                try:
                    sys.modules[cum_path] = import_module(cum_path)
                except Exception:
                    logger.info(f'Setting enricher at {cum_path}')
                    sys.modules[cum_path] = module_class

        grouped_enrichers: dict[str, list[EnricherReference]] = defaultdict(list)

        for enricher in enrichers:
            grouped_enrichers[enricher.module].append(enricher)

        for module, values in grouped_enrichers.items():
            set_module(
                module, DynamicEnricher({enricher.attribute_name: enricher.enricher for enricher in values})
            )

    @staticmethod
    def from_definition(repo: RepoDefinition, feature_source: FeatureSource | None = None) -> 'FeatureStore':
        """Creates a feature store based on a repo definition
        A feature source can also be defined if wanted, otherwise will the batch source be used for reads

        ```
        repo_file: bytes = ...
        repo_def = RepoDefinition.from_json(repo_file)
        feature_store = FeatureStore.from_definition(repo_def)
        ```

        Args:
            repo (RepoDefinition): The definition to setup
            feature_source (FeatureSource | None, optional): The source to read from and potentially write to.

        Returns:
            FeatureStore: A ready to use feature store
        """
        source = feature_source or repo.online_source.feature_source(repo.feature_views)
        feature_views = {fv.name: fv for fv in repo.feature_views}
        combined_feature_views = {fv.name: fv for fv in repo.combined_feature_views}

        FeatureStore.register_enrichers(repo.enrichers)

        return FeatureStore(
            feature_views=feature_views,
            combined_feature_views=combined_feature_views,
            models={model.name: model for model in repo.models},
            feature_source=source,
        )

    def repo_definition(self) -> RepoDefinition:
        return RepoDefinition(
            metadata=RepoMetadata(datetime.utcnow(), 'feature_store_location.py'),
            feature_views=set(self.feature_views.values()),
            combined_feature_views=set(self.combined_feature_views.values()),
            models=set(self.models.values()),
            online_source=BatchOnlineSource(),
            enrichers=[],
        )

    @staticmethod
    async def from_reference_at_path(
        path: str = '.', reference_file: str = 'feature_store_location.py'
    ) -> 'FeatureStore':
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
        repo_def = await RepoDefinition.from_reference_at_path(path, reference_file)
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
        definition = await RepoDefinition.from_path(path)
        return FeatureStore.from_definition(definition)

    def features_for_request(
        self, requests: FeatureRequest, entities: dict[str, list] | RetrivalJob, feature_names: set[str]
    ) -> RetrivalJob:
        entity_request: RetrivalJob

        if isinstance(entities, dict):
            if requests.needs_event_timestamp and self.event_timestamp_column not in entities:
                raise ValueError(f'Missing {self.event_timestamp_column} in entities')

            entity_request = RetrivalJob.from_dict(entities, requests)
        else:
            entity_request = entities

        return FilterJob(
            feature_names,
            self.feature_source.features_for(entity_request, requests),
        )

    def features_for(self, entities: dict[str, list] | RetrivalJob, features: list[str]) -> RetrivalJob:

        feature_request = RawStringFeatureRequest(features=set(features))
        requests = self.requests_for(feature_request)

        feature_names = set()

        if requests.needs_event_timestamp:
            feature_names.add(self.event_timestamp_column)

        for view, feature_set in feature_request.grouped_features.items():
            if feature_set != {'*'}:
                feature_names.update(feature_set)
            else:
                for request in requests.needed_requests:
                    if view.name == request.location.name:
                        feature_names.update(request.all_feature_names)
        for request in requests.needed_requests:
            feature_names.update(request.entity_names)

        return self.features_for_request(requests, entities, feature_names)

    def model(self, name: str) -> 'ModelFeatureStore':
        model = self.models[name]
        return ModelFeatureStore(model, self)

    def event_triggers_for(self, feature_view: str) -> set[EventTrigger]:
        triggers = self.feature_views[feature_view].event_triggers or set()
        for model in self.models.values():
            for target in model.predictions_view.classification_targets:
                if target.event_trigger and target.estimating.location.location == feature_view:
                    triggers.add(target.event_trigger)
        return triggers

    @staticmethod
    def _requests_for(
        feature_request: RawStringFeatureRequest,
        feature_views: dict[str, CompiledFeatureView],
        combined_feature_views: dict[str, CompiledCombinedFeatureView],
    ) -> FeatureRequest:
        features = feature_request.grouped_features
        requests: list[RetrivalRequest] = []
        entity_names = set()
        needs_event_timestamp = False

        for location in feature_request.locations:
            feature_view_name = location.name
            if feature_view_name in combined_feature_views:
                cfv = combined_feature_views[feature_view_name]
                if len(features[location]) == 1 and list(features[location])[0] == '*':
                    sub_requests = cfv.request_all
                else:
                    sub_requests = cfv.requests_for(features[location])
                requests.extend(sub_requests.needed_requests)
                for request in sub_requests.needed_requests:
                    entity_names.update(request.entity_names)
                    if request.event_timestamp:
                        needs_event_timestamp = True

            elif feature_view_name in feature_views:
                feature_view = feature_views[feature_view_name]
                if len(features[location]) == 1 and list(features[location])[0] == '*':
                    sub_requests = feature_view.request_all
                else:
                    sub_requests = feature_view.request_for(features[location])
                requests.extend(sub_requests.needed_requests)
                for request in sub_requests.needed_requests:
                    entity_names.update(request.entity_names)
                    if request.event_timestamp:
                        needs_event_timestamp = True
            else:
                raise ValueError(
                    f'Unable to find: {feature_view_name}, '
                    f'availible views are: {combined_feature_views.keys()}, and: {feature_views.keys()}'
                )

        if needs_event_timestamp:
            entity_names.add('event_timestamp')

        return FeatureRequest(
            FeatureLocation.model('custom features'),
            feature_request.feature_names.union(entity_names),
            RetrivalRequest.combine(requests),
        )

    def requests_for(self, feature_request: RawStringFeatureRequest) -> FeatureRequest:
        return FeatureStore._requests_for(feature_request, self.feature_views, self.combined_feature_views)

    def feature_view(self, view: str) -> 'FeatureViewStore':
        if view in self.combined_feature_views:
            raise CombinedFeatureViewQuerying(
                'You are trying to get a combined feature view. ',
                'This is only possible through store.features_for(...), as of now.\n',
            )
        feature_view = self.feature_views[view]
        return FeatureViewStore(self, feature_view, self.event_triggers_for(view))

    def add_feature_view(self, feature_view: FeatureView) -> None:
        compiled_view = type(feature_view).compile()
        self.feature_views[compiled_view.name] = compiled_view
        if isinstance(self.feature_source, BatchFeatureSource):
            self.feature_source.sources[
                FeatureLocation.feature_view(compiled_view.name).identifier
            ] = compiled_view.batch_data_source

    def add_combined_feature_view(self, feature_view: CombinedFeatureView) -> None:
        compiled_view = type(feature_view).compile()
        self.combined_feature_views[compiled_view.name] = compiled_view

    def add_model(self, model: Model) -> None:
        compiled_model = type(model).compile()
        self.models[compiled_model.name] = compiled_model

    def with_source(self, source: OnlineSource | None = None) -> 'FeatureStore':
        """
        Creates a new instance of a feature store, but changes where to fetch the features from

        ```
        store = # Load the store
        redis_store = store.with_source(redis.online_source())
        batch_source = redis_store.with_source()
        ```

        Args:
            source (OnlineSource): The source to fetch from, None will lead to using the batch source

        Returns:
            FeatureStore: A new feature store instance
        """
        online_source = source or BatchOnlineSource()
        return FeatureStore(
            feature_views=self.feature_views,
            combined_feature_views=self.combined_feature_views,
            models=self.models,
            feature_source=online_source.feature_source(set(self.feature_views.values())),
        )

    def offline_store(self) -> 'FeatureStore':
        return self.with_source()

    def model_features_for(self, view_name: str) -> set[str]:
        all_model_features: set[str] = set()
        for model in self.models.values():
            all_model_features.update(
                {feature.name for feature in model.features if feature.location.name == view_name}
            )
        return all_model_features


@dataclass
class ModelFeatureStore:

    model: ModelSchema
    store: FeatureStore

    @property
    def raw_string_features(self) -> set[str]:
        return {f'{feature.location.identifier}:{feature.name}' for feature in self.model.features}

    @property
    def request(self) -> FeatureRequest:
        return self.store.requests_for(RawStringFeatureRequest(self.raw_string_features))

    def features_for(self, entities: dict[str, list] | RetrivalJob) -> RetrivalJob:
        request = self.request
        features = self.raw_string_features
        return self.store.features_for(entities, list(features)).filter(request.features_to_include)

    def with_target(self) -> 'SupervisedModelFeatureStore':
        return SupervisedModelFeatureStore(self.model, self.store)

    def cached_at(self, location: DataFileReference) -> RetrivalJob:
        from aligned.local.job import FileFullJob

        features = {f'{feature.location.identifier}:{feature.name}' for feature in self.model.features}
        request = self.store.requests_for(RawStringFeatureRequest(features))

        return FileFullJob(location, RetrivalRequest.unsafe_combine(request.needed_requests)).filter(
            request.features_to_include
        )

    def process_features(self, input: RetrivalJob | dict[str, list]) -> RetrivalJob:
        request = self.request

        if isinstance(input, RetrivalJob):
            job = input.filter(request.features_to_include)
        elif isinstance(input, dict):
            job = RetrivalJob.from_dict(input, request=request.needed_requests)
        else:
            raise ValueError(f'features must be a dict or a RetrivalJob, was {type(input)}')

        return (
            job.ensure_types(request.needed_requests)
            .derive_features(request.needed_requests)
            .filter(request.features_to_include)
        )


@dataclass
class SupervisedModelFeatureStore:

    model: ModelSchema
    store: FeatureStore

    def features_for(self, entities: dict[str, list] | RetrivalJob) -> SupervisedJob:
        feature_refs = self.model.features
        features = {f'{feature.location.identifier}:{feature.name}' for feature in feature_refs}
        target_features = {
            f'{feature.estimating.location.identifier}:{feature.estimating.name}'
            for feature in self.model.predictions_view.classification_targets
        }
        targets = {feature.estimating.name for feature in self.model.predictions_view.classification_targets}
        request = self.store.requests_for(RawStringFeatureRequest(features))
        target_request = self.store.requests_for(
            RawStringFeatureRequest(target_features)
        ).without_event_timestamp(name_sufix='target')

        total_request = FeatureRequest(
            FeatureLocation.model(self.model.name),
            request.features_to_include.union(target_request.features_to_include),
            request.needed_requests + target_request.needed_requests,
        )
        job = self.store.features_for_request(total_request, entities, total_request.features_to_include)
        return SupervisedJob(
            job,
            target_columns=targets,
        )


@dataclass
class FeatureViewStore:

    store: FeatureStore
    view: CompiledFeatureView
    event_triggers: set[EventTrigger] = field(default_factory=set)
    feature_filter: set[str] | None = field(default=None)
    only_write_model_features: bool = field(default=False)

    @property
    def request(self) -> RetrivalRequest:
        if self.only_write_model_features:
            features_in_models = self.store.model_features_for(self.view.name)
            return self.view.request_for(features_in_models).needed_requests[0]
        else:
            return self.view.request_all.needed_requests[0]

    @property
    def source(self) -> FeatureSource:
        return self.store.feature_source

    def with_optimised_write(self) -> 'FeatureViewStore':
        features_in_models = self.store.model_features_for(self.view.name)
        return self.select(features_in_models)

    def all(self, limit: int | None = None) -> RetrivalJob:
        if not isinstance(self.source, RangeFeatureSource):
            raise ValueError(f'The source ({self.source}) needs to conform to RangeFeatureSource')

        request = self.view.request_all
        if self.feature_filter:
            request = self.view.request_for(self.feature_filter)

        job = (
            self.source.all_for(request, limit)
            .ensure_types(request.needed_requests)
            .derive_features(request.needed_requests)
        )
        if self.feature_filter:
            return FilterJob(include_features=self.feature_filter, job=job)
        else:
            return job

    def between_dates(self, start_date: datetime, end_date: datetime) -> RetrivalJob:
        if not isinstance(self.source, RangeFeatureSource):
            raise ValueError(
                f'The source needs to conform to RangeFeatureSource, you got a {type(self.source)}'
            )

        if self.feature_filter:
            request = self.view.request_for(self.feature_filter)
            return FilterJob(self.feature_filter, self.source.all_between(start_date, end_date, request))

        request = self.view.request_all
        return self.source.all_between(start_date, end_date, request)

    def previous(self, days: int = 0, minutes: int = 0, seconds: int = 0) -> RetrivalJob:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days, minutes=minutes, seconds=seconds)
        return self.between_dates(start_date, end_date)

    def features_for(self, entities: dict[str, list] | RetrivalJob) -> RetrivalJob:

        request = self.view.request_all
        if self.feature_filter:
            request = self.view.request_for(self.feature_filter)

        if isinstance(entities, dict):
            entity_job = RetrivalJob.from_dict(entities, request)
        elif isinstance(entities, RetrivalJob):
            entity_job = entities
        else:
            raise ValueError(f'entities must be a dict or a RetrivalJob, was {type(entities)}')

        job = self.source.features_for(entity_job, request)
        if self.feature_filter:
            return job.filter(self.feature_filter)
        else:
            return job

    def select(self, features: set[str]) -> 'FeatureViewStore':
        return FeatureViewStore(self.store, self.view, self.event_triggers, features)

    @property
    def write_input(self) -> set[str]:
        features = set()
        for request in self.view.request_all.needed_requests:
            features.update(request.all_required_feature_names)
            features.update(request.entity_names)
            if event_timestamp := request.event_timestamp:
                features.add(event_timestamp.name)
        return features

    async def write(self, values: dict[str, list[Any]]) -> None:
        from aligned import FileSource
        from aligned.data_file import DataFileReference
        from aligned.schemas.derivied_feature import AggregateOver

        request = self.view.request_all.needed_requests[0]
        job = (
            RetrivalJob.from_dict(values, request)
            .validate_entites()
            .fill_missing_columns()
            .ensure_types([request])
        )

        aggregations = request.aggregate_over()
        if aggregations:
            checkpoints: dict[AggregateOver, DataFileReference] = {}

            for aggregation in aggregations.keys():
                name = f'{self.view.name}_agg'

                if aggregation.window:
                    name += f'_{aggregation.window.time_window.total_seconds()}'

                if aggregation.condition:
                    name += f'_{aggregation.condition.name}'

                checkpoints[aggregation] = FileSource.parquet_at(name)

            job = StreamAggregationJob(job, checkpoints)

        await self.batch_write(job)

    def process_input(self, values: dict[str, list[Any]]) -> RetrivalJob:

        request = self.view.request_all.needed_requests[0]

        job = RetrivalJob.from_dict(values, request)

        return job.fill_missing_columns().ensure_types([request]).derive_features([request])

    async def batch_write(self, values: dict[str, list[Any]] | RetrivalJob) -> None:

        if not isinstance(self.source, WritableFeatureSource):
            logger.info('Feature Source is not writable')
            return

        # As it is a feature view, should it only contain one request
        request = self.request

        core_job: RetrivalJob

        if isinstance(values, RetrivalJob):
            core_job = values
        elif isinstance(values, dict):
            core_job = RetrivalJob.from_dict(values, request)
        else:
            raise ValueError(f'values must be a dict or a RetrivalJob, was {type(values)}')

        job = (
            core_job.derive_features([request])
            .listen_to_events(self.event_triggers)
            .update_vector_index(self.view.indexes)
        )

        if self.feature_filter:
            logger.info(f'Only writing features used by models: {self.feature_filter}')
            job = job.filter(self.feature_filter)

        await self.source.write(job, [request])
