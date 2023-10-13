from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import import_module
from typing import Any

from prometheus_client import Histogram

from aligned.compiler.model import ModelContract
from aligned.data_file import DataFileReference
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.enricher import Enricher
from aligned.feature_source import (
    BatchFeatureSource,
    FeatureSource,
    FeatureSourceFactory,
    RangeFeatureSource,
    WritableFeatureSource,
)
from aligned.feature_view.combined_view import CombinedFeatureView, CompiledCombinedFeatureView
from aligned.feature_view.feature_view import FeatureView
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.retrival_job import FilterJob, RetrivalJob, StreamAggregationJob, SupervisedJob
from aligned.schemas.feature import FeatureLocation, Feature
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.schemas.model import EventTrigger
from aligned.schemas.model import Model as ModelSchema
from aligned.schemas.repo_definition import EnricherReference, RepoDefinition, RepoMetadata

logger = logging.getLogger(__name__)

feature_view_write_time = Histogram(
    'feature_view_write_time',
    'The time used to write data related to a feature view',
    labelnames=['feature_view'],
)


@dataclass
class SourceRequest:
    """
    Represent a request to a source.
    This can be used validate the sources.
    """

    location: FeatureLocation
    source: BatchDataSource
    request: RetrivalRequest


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
    def experimental() -> FeatureStore:
        return FeatureStore.from_definition(
            RepoDefinition(
                metadata=RepoMetadata(created_at=datetime.utcnow(), name='experimental'),
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
    def from_definition(repo: RepoDefinition, feature_source: FeatureSource | None = None) -> FeatureStore:
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
        feature_views = {fv.name: fv for fv in repo.feature_views}
        combined_feature_views = {fv.name: fv for fv in repo.combined_feature_views}

        FeatureStore.register_enrichers(repo.enrichers)
        sources = {
            FeatureLocation.feature_view(view.name).identifier: view.batch_data_source
            for view in repo.feature_views
        } | {
            FeatureLocation.model(model.name).identifier: model.predictions_view.source
            for model in repo.models
            if model.predictions_view.source is not None
        }

        return FeatureStore(
            feature_views=feature_views,
            combined_feature_views=combined_feature_views,
            models={model.name: model for model in repo.models},
            feature_source=BatchFeatureSource(sources),
        )

    def repo_definition(self) -> RepoDefinition:
        return RepoDefinition(
            metadata=RepoMetadata(datetime.utcnow(), 'feature_store_location.py'),
            feature_views=set(self.feature_views.values()),
            combined_feature_views=set(self.combined_feature_views.values()),
            models=set(self.models.values()),
            enrichers=[],
        )

    @staticmethod
    async def from_reference_at_path(
        path: str = '.', reference_file: str = 'feature_store_location.py'
    ) -> FeatureStore:
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
    async def from_dir(path: str = '.') -> FeatureStore:
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

        return self.feature_source.features_for(entity_request, requests).filter(feature_names)

    def features_for(self, entities: dict[str, list] | RetrivalJob, features: list[str]) -> RetrivalJob:
        """
        Returns a set of features given a set of entities.

        ```python
        entities = { "user_id": [1, 2, 3, ...] }
        job = store.features_for(entities, features=["user:time_since_last_login", ...])
        data = await job.to_pandas()
        ```

        Args:
            entities (dict[str, list] | RetrivalJob): The entities to load data for
            features (list[str]): A list of features to load. Use the format (<feature_view>:<feature>)

        Returns:
            RetrivalJob: A job that knows how to fetch the features
        """

        feature_request = RawStringFeatureRequest(features=set(features))
        requests = self.requests_for(feature_request)

        feature_names = set()

        if requests.needs_event_timestamp:
            feature_names.add(self.event_timestamp_column)
            if isinstance(entities, dict) and self.event_timestamp_column not in entities:
                length = len(list(entities.values())[0])
                entities[self.event_timestamp_column] = [datetime.utcnow()] * length

        for view, feature_set in feature_request.grouped_features.items():
            if feature_set != {'*'}:
                feature_names.update(feature_set)
            else:
                for request in requests.needed_requests:
                    if view.name == request.location.name:
                        feature_names.update(request.all_feature_names)

        for request_index in range(len(requests.needed_requests)):
            request = requests.needed_requests[request_index]
            feature_names.update(request.entity_names)

            if isinstance(entities, dict):
                # Do not load the features if they already exist as an entity
                request.features = {feature for feature in request.features if feature.name not in entities}
            if len(request.features) == 0 and request.location.location != 'combined_view':
                request.derived_features = set()

        return self.features_for_request(requests, entities, feature_names)

    def model(self, name: str) -> ModelFeatureStore:
        """
        Selects a model for easy of use.

        Returns:
            ModelFeatureStore: A new store that containes the selected model
        """
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
        models: dict[str, ModelSchema],
    ) -> FeatureRequest:
        features = feature_request.grouped_features
        requests: list[RetrivalRequest] = []
        entity_names = set()
        needs_event_timestamp = False

        for location in feature_request.locations:
            location_name = location.name
            if location.location == 'model':
                model = models[location_name]
                view = model.predictions_view
                if len(features[location]) == 1 and list(features[location])[0] == '*':
                    request = view.request(location_name)
                else:
                    request = view.request_for(features[location], location_name)
                requests.append(request)
                entity_names.update(request.entity_names)
                if request.event_timestamp:
                    needs_event_timestamp = True

            elif location_name in combined_feature_views:
                cfv = combined_feature_views[location_name]
                if len(features[location]) == 1 and list(features[location])[0] == '*':
                    sub_requests = cfv.request_all
                else:
                    sub_requests = cfv.requests_for(features[location])
                requests.extend(sub_requests.needed_requests)
                for request in sub_requests.needed_requests:
                    entity_names.update(request.entity_names)
                    if request.event_timestamp:
                        needs_event_timestamp = True

            elif location_name in feature_views:
                feature_view = feature_views[location_name]
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
                    f'Unable to find: {location_name}, '
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
        return FeatureStore._requests_for(
            feature_request, self.feature_views, self.combined_feature_views, self.models
        )

    def feature_view(self, view: str) -> FeatureViewStore:
        """
        Selects a feature view based on a name.

        From here can you query the feature view for features.

        ```python
        data = await store.feature_view('my_view').all(limit=10).to_pandas()
        ```

        Args:
            view (str): The name of the feature view

        Raises:
            CombinedFeatureViewQuerying: If the name is a combined feature view

        Returns:
            FeatureViewStore: The selected feature view ready for querying
        """
        if view in self.combined_feature_views:
            return FeatureViewStore(self, self.combined_feature_views[view], set())
        feature_view = self.feature_views[view]
        return FeatureViewStore(self, feature_view, self.event_triggers_for(view))

    def add_view(self, view: CompiledFeatureView) -> None:
        """
        Compiles and adds the feature view to the store

        ```python
        @feature_view(...)
        class MyFeatureView:

            id = Int32().as_entity()

            my_feature = String()

        store.add_compiled_view(MyFeatureView.compile())
        ```

        Args:
            view (CompiledFeatureView): The feature view to add
        """
        self.add_compiled_view(view)

    def add_compiled_view(self, view: CompiledFeatureView) -> None:
        """
        Compiles and adds the feature view to the store

        ```python
        @feature_view(...)
        class MyFeatureView:

            id = Int32().as_entity()

            my_feature = String()

        store.add_compiled_view(MyFeatureView.compile())
        ```

        Args:
            view (CompiledFeatureView): The feature view to add
        """
        self.feature_views[view.name] = view
        if isinstance(self.feature_source, BatchFeatureSource):
            self.feature_source.sources[
                FeatureLocation.feature_view(view.name).identifier
            ] = view.batch_data_source

    def add_feature_view(self, feature_view: FeatureView) -> None:
        self.add_compiled_view(feature_view.compile_instance())

    def add_combined_feature_view(self, feature_view: CombinedFeatureView) -> None:
        compiled_view = type(feature_view).compile()
        self.combined_feature_views[compiled_view.name] = compiled_view

    def add_combined_view(self, compiled_view: CompiledCombinedFeatureView) -> None:
        self.combined_feature_views[compiled_view.name] = compiled_view

    def add_model(self, model: ModelContract) -> None:
        """
        Compiles and adds the model to the store

        Args:
            model (Model): The model to add
        """
        compiled_model = type(model).compile()
        self.models[compiled_model.name] = compiled_model

    def with_source(self, source: FeatureSource | FeatureSourceFactory | None = None) -> FeatureStore:
        """
        Creates a new instance of a feature store, but changes where to fetch the features from

        ```
        store = # Load the store
        redis_store = store.with_source(redis)
        batch_source = redis_store.with_source()
        ```

        Args:
            source (FeatureSource): The source to fetch from, None will lead to using the batch source

        Returns:
            FeatureStore: A new feature store instance
        """
        if isinstance(source, FeatureSourceFactory):
            feature_source = source.feature_source()
        else:
            sources = {
                FeatureLocation.feature_view(view.name).identifier: view.batch_data_source
                for view in set(self.feature_views.values())
            } | {
                FeatureLocation.model(model.name).identifier: model.predictions_view.source
                for model in set(self.models.values())
                if model.predictions_view.source is not None
            }
            feature_source = source or BatchFeatureSource(sources=sources)

        return FeatureStore(
            feature_views=self.feature_views,
            combined_feature_views=self.combined_feature_views,
            models=self.models,
            feature_source=feature_source,
        )

    def offline_store(self) -> FeatureStore:
        """
        Will set the source to the defined batch sources.

        Returns:
            FeatureStore: A new feature store that loads features from the batch sources
        """
        return self.with_source()

    def use_application_sources(self) -> FeatureStore:
        """
        Selects features from the application source if added.
        Otherwise, the we will default back to the batch source.

        Returns:
            FeatureStore: A new feature store that loads features from the application source
        """
        sources = {
            FeatureLocation.feature_view(view.name).identifier: view.application_source
            or view.batch_data_source
            for view in set(self.feature_views.values())
        } | {
            FeatureLocation.model(model.name).identifier: model.predictions_view.source
            for model in set(self.models.values())
            if model.predictions_view.source is not None
        }
        return FeatureStore(
            feature_views=self.feature_views,
            combined_feature_views=self.combined_feature_views,
            models=self.models,
            feature_source=BatchFeatureSource(sources=sources),
        )

    def model_features_for(self, view_name: str) -> set[str]:
        all_model_features: set[str] = set()
        for model in self.models.values():
            all_model_features.update(
                {feature.name for feature in model.features if feature.location.name == view_name}
            )
        return all_model_features

    def views_with_config(self, config: Any) -> list[SourceRequest]:
        """
        Returns the feature views where the config match.

        ```python
        source = PostgreSQLConfig(env_var='SOURCE_URL')
        store.views_with_conifg(source)
        ```

        Args:
            config (Any): The config to find views for

        Returns:
            list[SourceRequest]: A list of data sources, the request and it's location
        """
        views: list[SourceRequest] = []
        for view in self.feature_views.values():
            request = view.request_all.needed_requests[0]
            if view.batch_data_source.contains_config(config):
                views.append(
                    SourceRequest(FeatureLocation.feature_view(view.name), view.batch_data_source, request)
                )

            if view.application_source and view.application_source.contains_config(config):
                views.append(
                    SourceRequest(FeatureLocation.feature_view(view.name), view.application_source, request)
                )
        return views


@dataclass
class ModelFeatureStore:

    model: ModelSchema
    store: FeatureStore

    @property
    def location(self) -> FeatureLocation:
        return FeatureLocation.model(self.model.name)

    def raw_string_features(self, except_features: set[str]) -> set[str]:
        return {
            f'{feature.location.identifier}:{feature.name}'
            for feature in self.model.features
            if feature.name not in except_features
        }

    def request(self, except_features: set[str] | None = None) -> FeatureRequest:
        return self.store.requests_for(
            RawStringFeatureRequest(self.raw_string_features(except_features or set()))
        )

    def needed_entities(self) -> set[Feature]:
        return self.request().request_result.entities

    def features_for(self, entities: dict[str, list] | RetrivalJob) -> RetrivalJob:
        """Returns the features for the given entities

        ```python
        store = await FileSource.json_at("features-latest.json").feature_store()

        df = store.model("titanic")\\
            .features_for({"passenger_id": [1, 2, 3]})\\
            .to_polars()

        print(df.collect())
        >>> ┌──────────────┬───────┬─────────┬─────────────────────┬──────────────┐
        >>> │ passenger_id ┆ is_mr ┆ is_male ┆ constant_filled_age ┆ has_siblings │
        >>> │ ---          ┆ ---   ┆ ---     ┆ ---                 ┆ ---          │
        >>> │ i32          ┆ bool  ┆ bool    ┆ f64                 ┆ bool         │
        >>> ╞══════════════╪═══════╪═════════╪═════════════════════╪══════════════╡
        >>> │ 1            ┆ true  ┆ true    ┆ 22.0                ┆ true         │
        >>> │ 2            ┆ false ┆ false   ┆ 38.0                ┆ true         │
        >>> │ 3            ┆ false ┆ false   ┆ 26.0                ┆ false        │
        >>> └──────────────┴───────┴─────────┴─────────────────────┴──────────────┘
        ```

        Args:
            entities (dict[str, list] | RetrivalJob): The entities to fetch features for

        Returns:
            RetrivalJob: A retrival job that can be used to fetch the features
        """
        request = self.request()
        if isinstance(entities, dict):
            features = self.raw_string_features(set(entities.keys()))
        else:
            features = self.raw_string_features(set())

        job = self.store.features_for(entities, list(features)).with_request(request.needed_requests)

        if isinstance(entities, dict):
            subset_request = self.request(set(entities.keys()))

            if subset_request.request_result.feature_columns != request.request_result.feature_columns:
                job = job.derive_features(request.needed_requests)

        return job.filter(request.features_to_include)

    async def freshness(self) -> dict[FeatureLocation, datetime]:
        from aligned.schemas.feature import EventTimestamp

        locs: dict[FeatureLocation, EventTimestamp] = {}

        for req in self.request().needed_requests:
            if req.event_timestamp:
                locs[req.location]

        return await self.store.feature_source.freshness_for(locs)

    def with_labels(self) -> SupervisedModelFeatureStore:
        """Will also load the labels for the model

        ```python
        store = await FileSource.json_at("features-latest.json").feature_store()

        data = store.model("titanic")\\
            .with_labels()\\
            .features_for({"passenger_id": [1, 2, 3]})\\
            .to_polars()

        print(data.labels.collect(), data.input.collect())
        >>> ┌──────────┐ ┌───────┬─────────┬─────────────────────┬──────────────┐
        >>> │ survived │ │ is_mr ┆ is_male ┆ constant_filled_age ┆ has_siblings │
        >>> │ ---      │ │ ---   ┆ ---     ┆ ---                 ┆ ---          │
        >>> │ bool     │ │ bool  ┆ bool    ┆ f64                 ┆ bool         │
        >>> ╞══════════╡ ╞═══════╪═════════╪═════════════════════╪══════════════╡
        >>> │ false    │ │ true  ┆ true    ┆ 22.0                ┆ true         │
        >>> │ true     │ │ false ┆ false   ┆ 38.0                ┆ true         │
        >>> │ true     │ │ false ┆ false   ┆ 26.0                ┆ false        │
        >>> └──────────┘ └───────┴─────────┴─────────────────────┴──────────────┘
        ```

        Returns:
            SupervisedModelFeatureStore: A new queryable feature store
        """
        return SupervisedModelFeatureStore(self.model, self.store)

    def cached_at(self, location: DataFileReference) -> RetrivalJob:
        """Loads the model features from a pre computed location

        ```python
        from aligned import FileSource

        store = await FileSource.json_at("features-latest.json").feature_store()

        cached_features = FileSource.parquet_at("titanic_features.parquet")

        df = store.model("titanic")\\
            .cached_at(cached_features)\\
            .to_polars()

        print(df.collect())
        >>> ┌──────────────┬───────┬─────────┬─────────────────────┬──────────────┐
        >>> │ passenger_id ┆ is_mr ┆ is_male ┆ constant_filled_age ┆ has_siblings │
        >>> │ ---          ┆ ---   ┆ ---     ┆ ---                 ┆ ---          │
        >>> │ i32          ┆ bool  ┆ bool    ┆ f64                 ┆ bool         │
        >>> ╞══════════════╪═══════╪═════════╪═════════════════════╪══════════════╡
        >>> │ 1            ┆ true  ┆ true    ┆ 22.0                ┆ true         │
        >>> │ 2            ┆ false ┆ false   ┆ 38.0                ┆ true         │
        >>> │ 3            ┆ false ┆ false   ┆ 26.0                ┆ false        │
        >>> └──────────────┴───────┴─────────┴─────────────────────┴──────────────┘
        ```

        Args:
            location (DataFileReference): _description_

        Returns:
            RetrivalJob: _description_
        """
        from aligned.local.job import FileFullJob

        features = {f'{feature.location.identifier}:{feature.name}' for feature in self.model.features}
        request = self.store.requests_for(RawStringFeatureRequest(features))

        return FileFullJob(location, RetrivalRequest.unsafe_combine(request.needed_requests)).filter(
            request.features_to_include
        )

    def process_features(self, input: RetrivalJob | dict[str, list]) -> RetrivalJob:
        request = self.request()

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

    def predictions_for(self, entities: dict[str, list] | RetrivalJob) -> RetrivalJob:

        location_id = self.location.identifier
        return self.store.features_for(entities, features=[f'{location_id}:*'])

    def all_predictions(self, limit: int | None = None) -> RetrivalJob:

        pred_view = self.model.predictions_view

        if pred_view.source is None:
            raise ValueError(
                'Model does not have a prediction source. '
                'This can be set in the metadata for a model contract.'
            )

        request = pred_view.request(self.model.name)
        return pred_view.source.all_data(request, limit=limit)

    def using_source(
        self, source: FeatureSource | FeatureSourceFactory | BatchDataSource
    ) -> ModelFeatureStore:

        model_source: FeatureSource | FeatureSourceFactory

        if isinstance(source, BatchDataSource):
            model_source = BatchFeatureSource({FeatureLocation.model(self.model.name).identifier: source})
        else:
            model_source = source

        return ModelFeatureStore(self.model, self.store.with_source(model_source))

    async def write_predictions(self, predictions: dict[str, list] | RetrivalJob) -> None:
        """
        Writes data to a source defined as a prediction source

        ```python
        @model_contract(
            name="taxi_eta",
            features=[...]
            predictions_source=FileSource.parquet_at("predictions.parquet")
        )
        class TaxiEta:
            trip_id = Int32().as_entity()

            duration = Int32()

        ...

        store = FeatureStore.from_dir(".")

        await store.model("taxi_eta").write_predictions({
            "trip_id": [1, 2, 3, ...],
            "duration": [20, 33, 42, ...]
        })
        ```
        """

        source: Any = self.store.feature_source

        if isinstance(source, BatchFeatureSource):
            location = FeatureLocation.model(self.model.name).identifier
            source = source.sources[location]

        if not isinstance(source, WritableFeatureSource):
            raise ValueError(f'The prediction source {type(source)} needs to be writable')

        write_job: RetrivalJob
        request = self.model.predictions_view.request(self.model.name)

        if isinstance(predictions, dict):
            write_job = RetrivalJob.from_dict(predictions, request)
        elif isinstance(predictions, RetrivalJob):
            write_job = predictions
        else:
            raise ValueError(f'Unable to write predictions of type {type(predictions)}')

        await source.write(write_job, [request])


@dataclass
class SupervisedModelFeatureStore:

    model: ModelSchema
    store: FeatureStore

    def features_for(self, entities: dict[str, list] | RetrivalJob) -> SupervisedJob:
        """Loads the features and labels for a model

        ```python
        store = await FileSource.json_at("features-latest.json").feature_store()

        data = store.model("titanic")\\
            .with_labels()\\
            .features_for({"passenger_id": [1, 2, 3]})\\
            .to_polars()

        print(data.labels.collect(), data.input.collect())
        >>> ┌──────────┐ ┌───────┬─────────┬─────────────────────┬──────────────┐
        >>> │ survived │ │ is_mr ┆ is_male ┆ constant_filled_age ┆ has_siblings │
        >>> │ ---      │ │ ---   ┆ ---     ┆ ---                 ┆ ---          │
        >>> │ bool     │ │ bool  ┆ bool    ┆ f64                 ┆ bool         │
        >>> ╞══════════╡ ╞═══════╪═════════╪═════════════════════╪══════════════╡
        >>> │ false    │ │ true  ┆ true    ┆ 22.0                ┆ true         │
        >>> │ true     │ │ false ┆ false   ┆ 38.0                ┆ true         │
        >>> │ true     │ │ false ┆ false   ┆ 26.0                ┆ false        │
        >>> └──────────┘ └───────┴─────────┴─────────────────────┴──────────────┘
        ```

        Args:
            entities (dict[str, list] | RetrivalJob): A dictionary of entity names to lists of entity values

        Returns:
            SupervisedJob: A object that will load the features and lables in your desired format
        """
        feature_refs = self.model.features
        features = {f'{feature.location.identifier}:{feature.name}' for feature in feature_refs}
        pred_view = self.model.predictions_view

        target_feature_refs = pred_view.labels_estimates_refs()
        target_features = {feature.identifier for feature in target_feature_refs}

        targets = set()
        if pred_view.classification_targets:
            targets = {feature.estimating.name for feature in pred_view.classification_targets}
        elif pred_view.regression_targets:
            targets = {feature.estimating.name for feature in pred_view.regression_targets}
        else:
            raise ValueError('Found no targets in the model')

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
            job.filter(total_request.features_to_include),
            target_columns=targets,
        )

    def predictions_for(self, entities: dict[str, list] | RetrivalJob) -> RetrivalJob:
        """Loads the predictions and labels / ground truths for a model

        ```python
        entities = {
            "trip_id": ["ea6b8d5d-62fd-4664-a112-4889ebfcdf2b", ...],
            "vendor_id": [2, ...],
        }
        preds = await store.model("taxi")\\
            .with_labels()\\
            .predictions_for(entities)\\
            .to_polars()

        print(preds.collect())
        >>> ┌──────────┬───────────┬────────────────────┬───────────────────────────────────┐
        >>> │ duration ┆ vendor_id ┆ predicted_duration ┆ trip_id                           │
        >>> │ ---      ┆ ---       ┆ ---                ┆ ---                               │
        >>> │ i64      ┆ i32       ┆ i64                ┆ str                               │
        >>> ╞══════════╪═══════════╪════════════════════╪═══════════════════════════════════╡
        >>> │ 408      ┆ 2         ┆ 500                ┆ ea6b8d5d-62fd-4664-a112-4889ebfc… │
        >>> │ 280      ┆ 1         ┆ 292                ┆ 64c4c94f-2a85-406f-86e6-082f1f7a… │
        >>> │ 712      ┆ 4         ┆ 689                ┆ 3258461f-6113-4c5e-864b-75a0dee8… │
        >>> └──────────┴───────────┴────────────────────┴───────────────────────────────────┘
        ```

        Args:
            entities (dict[str, list] | RetrivalJob): A dictionary of entity names to lists of entity values

        Returns:
            RetrivalJob: A object that will load the features and lables in your desired format
        """

        pred_view = self.model.predictions_view
        if pred_view.source is None:
            raise ValueError(
                'Model does not have a prediction source. '
                'This can be set in the metadata for a model contract.'
            )

        request = pred_view.request(self.model.name)

        target_features = pred_view.labels_estimates_refs()
        labels = pred_view.labels()
        target_features = {feature.identifier for feature in target_features}
        pred_features = {f'model:{self.model.name}:{feature.name}' for feature in labels}
        request = self.store.requests_for(RawStringFeatureRequest(pred_features))
        target_request = self.store.requests_for(
            RawStringFeatureRequest(target_features)
        ).without_event_timestamp(name_sufix='target')

        total_request = FeatureRequest(
            FeatureLocation.model(self.model.name),
            request.features_to_include.union(target_request.features_to_include),
            request.needed_requests + target_request.needed_requests,
        )
        return self.store.features_for_request(total_request, entities, total_request.features_to_include)


@dataclass
class FeatureViewStore:

    store: FeatureStore
    view: CompiledFeatureView
    event_triggers: set[EventTrigger] = field(default_factory=set)
    feature_filter: set[str] | None = field(default=None)

    @property
    def name(self) -> str:
        return self.view.name

    @property
    def request(self) -> RetrivalRequest:
        if self.feature_filter is not None:
            features_in_models = self.store.model_features_for(self.view.name)
            logger.info(f'Only processing model features: {features_in_models}')
            return self.view.request_for(features_in_models).needed_requests[0]
        else:
            return self.view.request_all.needed_requests[0]

    @property
    def source(self) -> FeatureSource:
        return self.store.feature_source

    def using_source(
        self, source: FeatureSource | FeatureSourceFactory | BatchDataSource
    ) -> FeatureViewStore:
        """
        Sets the source to load features from.

        ```python
        custom_source = PostgreSQLConfig.localhost("test")

        store = FeatureView.from_dir(".")

        features = await (store.feature_view("titanic")
            .using_source(custom_source)
            .all()
        )
        ```

        Args:
            source (BatchDataSource): The source to use

        Returns:
            A new `FeatureViewStore` that sends queries to the passed source
        """
        view_source: FeatureSource | FeatureSourceFactory

        if isinstance(source, BatchDataSource):
            view_source = BatchFeatureSource(
                {FeatureLocation.feature_view(self.view.name).identifier: source}
            )
        else:
            view_source = source

        return FeatureViewStore(
            self.store.with_source(view_source),
            view=self.view,
            event_triggers=self.event_triggers,
            feature_filter=self.feature_filter,
        )

    def with_optimised_write(self) -> FeatureViewStore:
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

    def select(self, features: set[str]) -> FeatureViewStore:
        logger.info(f'Selecting features {features}')
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
        from aligned.schemas.derivied_feature import AggregateOver

        request = self.view.request_all.needed_requests[0]
        if self.feature_filter is not None:
            logger.info(f'Filtering features to {self.feature_filter}')
            request = self.view.request_for(self.feature_filter)

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

        job = job.derive_features([request])

        if self.feature_filter:
            job = job.filter(self.feature_filter)

        await self.batch_write(job)

    def process_input(self, values: dict[str, list[Any]]) -> RetrivalJob:

        request = self.view.request_all.needed_requests[0]

        job = RetrivalJob.from_dict(values, request)

        return job.fill_missing_columns().ensure_types([request]).derive_features([request])

    async def batch_write(self, values: dict[str, list[Any]] | RetrivalJob) -> None:
        """Takes a set of features, computes the derived features, and store them in the source

        Args:
            values (dict[str, list[Any]] | RetrivalJob): The features to write

        Raises:
            ValueError: In case the inputed features are invalid
        """

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

        # job = (
        #     core_job.derive_features([request])
        #     .listen_to_events(self.event_triggers)
        #     .update_vector_index(self.view.indexes)
        # )
        job = core_job

        # if self.feature_filter:
        #     logger.info(f'Only writing features used by models: {self.feature_filter}')
        #     job = job.filter(self.feature_filter)

        with feature_view_write_time.labels(self.view.name).time():
            await self.source.write(job, job.retrival_requests)
