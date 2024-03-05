from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from importlib import import_module
from typing import Any, Union

from prometheus_client import Histogram

from aligned.compiler.model import ModelContractWrapper
from aligned.data_file import DataFileReference, upsert_on_column
from aligned.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aligned.enricher import Enricher
from aligned.exceptions import UnableToFindFileException
from aligned.feature_source import (
    BatchFeatureSource,
    FeatureSource,
    FeatureSourceFactory,
    RangeFeatureSource,
    WritableFeatureSource,
)
from aligned.feature_view.combined_view import CombinedFeatureView, CompiledCombinedFeatureView
from aligned.feature_view.feature_view import FeatureView, FeatureViewWrapper
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.retrival_job import (
    SelectColumnsJob,
    RetrivalJob,
    StreamAggregationJob,
    SupervisedJob,
    ConvertableToRetrivalJob,
)
from aligned.schemas.feature import FeatureLocation, Feature, FeatureReferance
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

FeatureSourceable = Union[FeatureSource, FeatureSourceFactory, None]


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
    def from_definition(repo: RepoDefinition) -> FeatureStore:
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
            FeatureLocation.feature_view(view.name).identifier: view.materialized_source
            if view.materialized_source
            else view.source
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
            metadata=RepoMetadata(datetime.utcnow(), name='feature_store_location.py'),
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
        self,
        requests: FeatureRequest,
        entities: ConvertableToRetrivalJob | RetrivalJob,
        feature_names: set[str],
    ) -> RetrivalJob:
        entity_request: RetrivalJob

        if isinstance(entities, RetrivalJob):
            entity_request = entities
        else:
            entity_request = RetrivalJob.from_convertable(entities, requests)

        return self.feature_source.features_for(entity_request, requests).select_columns(feature_names)

    def features_for(
        self,
        entities: ConvertableToRetrivalJob | RetrivalJob,
        features: list[str],
        event_timestamp_column: str | None = None,
    ) -> RetrivalJob:
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
        requests = self.requests_for(feature_request, event_timestamp_column)

        feature_names = set()

        if event_timestamp_column and requests.needs_event_timestamp:
            feature_names.add(event_timestamp_column)

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
            for target in model.predictions_view.classification_targets or set():
                if target.event_trigger and target.estimating.location.location == feature_view:
                    triggers.add(target.event_trigger)
        return triggers

    @staticmethod
    def _requests_for(
        feature_request: RawStringFeatureRequest,
        feature_views: dict[str, CompiledFeatureView],
        combined_feature_views: dict[str, CompiledCombinedFeatureView],
        models: dict[str, ModelSchema],
        event_timestamp_column: str | None = None,
    ) -> FeatureRequest:
        features = feature_request.grouped_features
        requests: list[RetrivalRequest] = []
        entity_names = set()

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

            elif location_name in combined_feature_views:
                cfv = combined_feature_views[location_name]
                if len(features[location]) == 1 and list(features[location])[0] == '*':
                    sub_requests = cfv.request_all
                else:
                    sub_requests = cfv.requests_for(features[location])
                requests.extend(sub_requests.needed_requests)
                for request in sub_requests.needed_requests:
                    entity_names.update(request.entity_names)

            elif location_name in feature_views:
                feature_view = feature_views[location_name]
                if len(features[location]) == 1 and list(features[location])[0] == '*':
                    sub_requests = feature_view.request_all
                else:
                    sub_requests = feature_view.request_for(features[location])
                requests.extend(sub_requests.needed_requests)
                for request in sub_requests.needed_requests:
                    entity_names.update(request.entity_names)
            else:
                raise ValueError(
                    f'Unable to find: {location_name}, '
                    f'availible views are: {combined_feature_views.keys()}, and: {feature_views.keys()}'
                )

        if event_timestamp_column:
            entity_names.add(event_timestamp_column)
            requests = [request.with_event_timestamp_column(event_timestamp_column) for request in requests]

        else:
            requests = [request.without_event_timestamp() for request in requests]

        return FeatureRequest(
            FeatureLocation.model('custom features'),
            feature_request.feature_names.union(entity_names),
            RetrivalRequest.combine(requests),
        )

    def requests_for(
        self, feature_request: RawStringFeatureRequest, event_timestamp_column: str | None = None
    ) -> FeatureRequest:
        return FeatureStore._requests_for(
            feature_request,
            self.feature_views,
            self.combined_feature_views,
            self.models,
            event_timestamp_column=event_timestamp_column,
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
            self.feature_source.sources[FeatureLocation.feature_view(view.name).identifier] = (
                view.materialized_source or view.source
            )

    def add_feature_view(self, feature_view: FeatureView | FeatureViewWrapper) -> None:
        if isinstance(feature_view, FeatureViewWrapper):
            self.add_compiled_view(feature_view.compile())
        else:
            self.add_compiled_view(feature_view.compile_instance())

    def add_combined_feature_view(self, feature_view: CombinedFeatureView) -> None:
        compiled_view = type(feature_view).compile()
        self.combined_feature_views[compiled_view.name] = compiled_view

    def add_combined_view(self, compiled_view: CompiledCombinedFeatureView) -> None:
        self.combined_feature_views[compiled_view.name] = compiled_view

    def add_model(self, model: ModelContractWrapper) -> None:
        """
        Compiles and adds the model to the store

        Args:
            model (Model): The model to add
        """
        compiled_model = model.compile()
        self.models[compiled_model.name] = compiled_model

    def add_compiled_model(self, model: ModelSchema) -> None:
        self.models[model.name] = model
        if isinstance(self.feature_source, BatchFeatureSource) and model.predictions_view.source:
            self.feature_source.sources[
                FeatureLocation.model(model.name).identifier
            ] = model.predictions_view.source

    def with_source(self, source: FeatureSourceable = None) -> FeatureStore:
        """
        Creates a new instance of a feature store, but changes where to fetch the features from

        ```
        store = await FeatureStore.from_dir(".")
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
        elif source is None:
            sources = {
                FeatureLocation.feature_view(view.name).identifier: view.source
                for view in set(self.feature_views.values())
            } | {
                FeatureLocation.model(model.name).identifier: model.predictions_view.source
                for model in set(self.models.values())
                if model.predictions_view.source is not None
            }
            feature_source = source or BatchFeatureSource(sources=sources)
        elif isinstance(source, FeatureSource):
            feature_source = source
        else:
            raise ValueError(
                'Setting a dedicated source needs to be either a FeatureSource, '
                f'or FeatureSourceFactory. Got: {type(source)}'
            )

        return FeatureStore(
            feature_views=self.feature_views,
            combined_feature_views=self.combined_feature_views,
            models=self.models,
            feature_source=feature_source,
        )

    def update_source_for(self, location: FeatureLocation | str, source: BatchDataSource) -> FeatureStore:
        if not isinstance(self.feature_source, BatchFeatureSource):
            raise ValueError(
                f'.update_source_for(...) needs a `BatchFeatureSource`, got {type(self.feature_source)}'
            )

        if isinstance(location, str):
            location = FeatureLocation.from_string(location)

        new_source = self.feature_source
        new_source.sources[location.identifier] = source

        return FeatureStore(
            feature_views=self.feature_views,
            combined_feature_views=self.combined_feature_views,
            models=self.models,
            feature_source=new_source,
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
            FeatureLocation.feature_view(view.name).identifier: view.application_source or view.source
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
                {
                    feature.name
                    for feature in model.features.default_features
                    if feature.location.name == view_name
                }
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
            if view.source.contains_config(config):
                views.append(SourceRequest(FeatureLocation.feature_view(view.name), view.source, request))

            if view.application_source and view.application_source.contains_config(config):
                views.append(
                    SourceRequest(FeatureLocation.feature_view(view.name), view.application_source, request)
                )
        return views

    def write_request_for(self, location: FeatureLocation) -> RetrivalRequest:

        if location.location == 'feature_view':
            return self.feature_views[location.name].request_all.needed_requests[0]
        elif location.location == 'model':
            return self.models[location.name].predictions_view.request('write')
        elif location.location == 'combined_view':
            raise NotImplementedError(
                'Have not implemented write requests for combined views. '
                'Please consider contributing and add a PR.'
            )
        else:
            raise ValueError(f"Unable to write to location: '{location}'.")

    async def insert_into(
        self, location: FeatureLocation | str, values: ConvertableToRetrivalJob | RetrivalJob
    ) -> None:

        if isinstance(location, str):
            used_location = FeatureLocation.from_string(location)
        elif isinstance(location, FeatureLocation):
            used_location = location
        else:
            raise ValueError(f'Location was of an unsupported type: {type(location)}')

        source: FeatureSource | BatchDataSource = self.feature_source

        if isinstance(source, BatchFeatureSource):
            source = source.sources[used_location.identifier]

        write_request = self.write_request_for(used_location)

        if not isinstance(values, RetrivalJob):
            values = RetrivalJob.from_convertable(values, write_request)

        if isinstance(source, WritableFeatureSource):
            await source.insert(values, [write_request])
        elif isinstance(source, DataFileReference):
            import polars as pl

            columns = write_request.all_returned_columns
            new_df = (await values.to_lazy_polars()).select(columns)

            try:
                existing_df = await source.to_lazy_polars()
                write_df = pl.concat([new_df, existing_df.select(columns)], how='vertical_relaxed')
            except UnableToFindFileException:
                write_df = new_df

            if isinstance(source, ColumnFeatureMappable):
                new_cols = source.feature_identifier_for(columns)

                mappings = dict(zip(columns, new_cols))
                write_df = write_df.rename(mappings)

            await source.write_polars(write_df)
        else:
            raise ValueError(f'The source {type(source)} do not support writes')

    async def upsert_into(
        self, location: FeatureLocation | str, values: ConvertableToRetrivalJob | RetrivalJob
    ) -> None:

        if isinstance(location, str):
            used_location = FeatureLocation.from_string(location)
        elif isinstance(location, FeatureLocation):
            used_location = location
        else:
            raise ValueError(f'Location was of an unsupported type: {type(location)}')

        source: FeatureSource | BatchDataSource = self.feature_source

        if isinstance(source, BatchFeatureSource):
            source = source.sources[used_location.identifier]

        write_request = self.write_request_for(used_location)

        if not isinstance(values, RetrivalJob):
            values = RetrivalJob.from_convertable(values, write_request)

        if isinstance(source, WritableFeatureSource):
            await source.upsert(values, [write_request])
        elif isinstance(source, DataFileReference):
            new_df = (await values.to_lazy_polars()).select(write_request.all_returned_columns)
            entities = list(write_request.entity_names)
            try:
                existing_df = await source.to_lazy_polars()
                write_df = upsert_on_column(entities, new_df, existing_df)
            except UnableToFindFileException:
                write_df = new_df
            await source.write_polars(write_df)
        else:
            raise ValueError(f'The source {type(source)} do not support writes')


@dataclass
class ModelFeatureStore:

    model: ModelSchema
    store: FeatureStore
    selected_version: str | None = None

    @property
    def location(self) -> FeatureLocation:
        return FeatureLocation.model(self.model.name)

    def raw_string_features(self, except_features: set[str]) -> set[str]:

        version = self.selected_version or self.model.features.default_version
        features = self.model.features.features_for(version)

        return {
            f'{feature.location.identifier}:{feature.name}'
            for feature in features
            if feature.name not in except_features
        }

    def using_version(self, version: str) -> ModelFeatureStore:
        return ModelFeatureStore(self.model, self.store, version)

    def request(
        self, except_features: set[str] | None = None, event_timestamp_column: str | None = None
    ) -> FeatureRequest:
        return self.store.requests_for(
            RawStringFeatureRequest(self.raw_string_features(except_features or set())),
            event_timestamp_column,
        )

    def needed_entities(self) -> set[Feature]:
        return self.request().request_result.entities

    def features_for(
        self, entities: ConvertableToRetrivalJob | RetrivalJob, event_timestamp_column: str | None = None
    ) -> RetrivalJob:
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
        request = self.request(event_timestamp_column=event_timestamp_column)
        if isinstance(entities, dict):
            features = self.raw_string_features(set(entities.keys()))
        else:
            features = self.raw_string_features(set())

        job = self.store.features_for(
            entities, list(features), event_timestamp_column=event_timestamp_column
        ).with_request(request.needed_requests)

        if isinstance(entities, dict):
            subset_request = self.request(set(entities.keys()), event_timestamp_column)

            if subset_request.request_result.feature_columns != request.request_result.feature_columns:
                job = job.derive_features(request.needed_requests)

        return job.select_columns(request.features_to_include)

    async def freshness(self) -> dict[FeatureLocation, datetime | None]:
        from aligned.schemas.feature import EventTimestamp

        locs: dict[FeatureLocation, EventTimestamp] = {}

        for req in self.request().needed_requests:
            if req.event_timestamp:
                locs[req.location] = req.event_timestamp

        return await self.store.feature_source.freshness_for(locs)

    def with_labels(self, label_refs: set[FeatureReferance] | None = None) -> SupervisedModelFeatureStore:
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
        return SupervisedModelFeatureStore(
            self.model,
            self.store,
            label_refs or self.model.predictions_view.labels_estimates_refs(),
            self.selected_version,
        )

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

        references = self.model.feature_references(self.selected_version)
        features = {f'{feature.location.identifier}:{feature.name}' for feature in references}
        request = self.store.requests_for(RawStringFeatureRequest(features))

        return FileFullJob(location, RetrivalRequest.unsafe_combine(request.needed_requests)).select_columns(
            request.features_to_include
        )

    def process_features(self, input: RetrivalJob | ConvertableToRetrivalJob) -> RetrivalJob:
        request = self.request()

        if isinstance(input, RetrivalJob):
            job = input.select_columns(request.features_to_include)
        else:
            job = RetrivalJob.from_convertable(input, request=request.needed_requests)

        return (
            job.ensure_types(request.needed_requests)
            .derive_features(request.needed_requests)
            .select_columns(request.features_to_include)
        )

    def predictions_for(
        self, entities: ConvertableToRetrivalJob | RetrivalJob, event_timestamp_column: str | None = None
    ) -> RetrivalJob:

        location_id = self.location.identifier
        return self.store.features_for(
            entities, features=[f'{location_id}:*'], event_timestamp_column=event_timestamp_column
        )

    def predictions_between(self, start_date: datetime, end_date: datetime) -> RetrivalJob:

        selected_source = self.store.feature_source

        if not isinstance(selected_source, BatchFeatureSource):
            raise ValueError(
                f'Unable to load all predictions for selected feature source {type(selected_source)}'
            )

        location = FeatureLocation.model(self.model.name)
        if location.identifier not in selected_source.sources:
            raise ValueError(
                f'Unable to find source for {location.identifier}. Either set through a `prediction_source`'
                'in the model contract, or use the `using_source` method on the store object.'
            )

        source = selected_source.sources[location.identifier]
        request = self.model.predictions_view.request(self.model.name)

        return source.all_between_dates(request, start_date, end_date).select_columns(
            set(request.all_returned_columns)
        )

    def all_predictions(self, limit: int | None = None) -> RetrivalJob:

        selected_source = self.store.feature_source

        if not isinstance(selected_source, BatchFeatureSource):
            raise ValueError(
                f'Unable to load all predictions for selected feature source {type(selected_source)}'
            )

        location = FeatureLocation.model(self.model.name)
        if location.identifier not in selected_source.sources:
            raise ValueError(
                f'Unable to find source for {location.identifier}. Either set through a `prediction_source`'
                'in the model contract, or use the `using_source` method on the store object.'
            )

        source = selected_source.sources[location.identifier]
        request = self.model.predictions_view.request(self.model.name)

        return source.all_data(request, limit=limit).select_columns(set(request.all_returned_columns))

    def using_source(self, source: FeatureSourceable | BatchDataSource) -> ModelFeatureStore:

        model_source: FeatureSourceable

        if isinstance(source, BatchDataSource):
            model_source = BatchFeatureSource({FeatureLocation.model(self.model.name).identifier: source})
        else:
            model_source = source

        return ModelFeatureStore(self.model, self.store.with_source(model_source))

    def depends_on(self) -> set[FeatureLocation]:
        """
        Returns the views and models that the model depend on to compute it's output.

        ```python
        @feature_view(name="passenger", ...)
        class Passenger:
            passenger_id = Int32().as_entity()

            age = Float()

        @feature_view(name="location", ...)
        class Location:
            location_id = String().as_entity()

            location_area = Float()


        @model_contract(name="some_model", ...)
        class SomeModel:
            some_id = String().as_entity()

            some_computed_metric = Int32()

        @model_contract(
            name="new_model",
            features=[
                Passenger().age,
                Location().location_area,
                SomeModel().some_computed_metric
            ]
        )
        class NewModel:
            ...

        print(store.model("new_model").depends_on())
        >>> {
        >>>     FeatureLocation(location="feature_view", name="passenger"),
        >>>     FeatureLocation(location="feature_view", name="location"),
        >>>     FeatureLocation(location="model", name="some_model")
        >>> }

        ```
        """
        return {req.location for req in self.request().needed_requests}

    async def upsert_predictions(self, predictions: ConvertableToRetrivalJob | RetrivalJob) -> None:
        """
        Upserts data to a source defined as a prediction source

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

        await store.model("taxi_eta").upsert_predictions({
            "trip_id": [1, 2, 3, ...],
            "duration": [20, 33, 42, ...]
        })
        ```
        """
        await self.store.upsert_into(FeatureLocation.model(self.model.name), predictions)

    async def insert_predictions(self, predictions: ConvertableToRetrivalJob | RetrivalJob) -> None:
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

        await store.model("taxi_eta").insert_predictions({
            "trip_id": [1, 2, 3, ...],
            "duration": [20, 33, 42, ...]
        })
        ```
        """
        await self.store.insert_into(FeatureLocation.model(self.model.name), predictions)


@dataclass
class SupervisedModelFeatureStore:

    model: ModelSchema
    store: FeatureStore
    labels_estimates_refs: set[FeatureReferance]

    selected_version: str | None = None

    def features_for(
        self,
        entities: ConvertableToRetrivalJob | RetrivalJob,
        event_timestamp_column: str | None = None,
        target_event_timestamp_column: str | None = None,
    ) -> SupervisedJob:
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
        feature_refs = self.model.feature_references(self.selected_version)
        features = {f'{feature.location.identifier}:{feature.name}' for feature in feature_refs}
        pred_view = self.model.predictions_view

        target_feature_refs = self.labels_estimates_refs
        target_features = {feature.identifier for feature in target_feature_refs}

        targets = set()
        if pred_view.classification_targets:
            targets = {feature.estimating.name for feature in pred_view.classification_targets}
        elif pred_view.regression_targets:
            targets = {feature.estimating.name for feature in pred_view.regression_targets}
        else:
            raise ValueError('Found no targets in the model')

        if event_timestamp_column == target_event_timestamp_column:
            request = self.store.requests_for(
                RawStringFeatureRequest(features.union(target_features)),
                event_timestamp_column=event_timestamp_column,
            )
            job = self.store.features_for_request(request, entities, request.features_to_include)
            return SupervisedJob(
                job.select_columns(request.features_to_include),
                target_columns=targets,
            )

        request = self.store.requests_for(
            RawStringFeatureRequest(features), event_timestamp_column=event_timestamp_column
        )
        target_request = self.store.requests_for(
            RawStringFeatureRequest(target_features), event_timestamp_column=target_event_timestamp_column
        ).with_sufix('target')

        total_request = FeatureRequest(
            FeatureLocation.model(self.model.name),
            request.features_to_include.union(target_request.features_to_include),
            request.needed_requests + target_request.needed_requests,
        )
        job = self.store.features_for_request(total_request, entities, total_request.features_to_include)
        return SupervisedJob(
            job.select_columns(total_request.features_to_include),
            target_columns=targets,
        )

    def predictions_for(
        self,
        entities: ConvertableToRetrivalJob | RetrivalJob,
        event_timestamp_column: str | None = None,
        target_event_timestamp_column: str | None = None,
    ) -> RetrivalJob:
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
        target_features = {feature.identifier for feature in target_features}

        labels = pred_view.labels()
        pred_features = {f'model:{self.model.name}:{feature.name}' for feature in labels}
        request = self.store.requests_for(
            RawStringFeatureRequest(pred_features), event_timestamp_column=event_timestamp_column
        )
        target_request = self.store.requests_for(
            RawStringFeatureRequest(target_features),
            event_timestamp_column=target_event_timestamp_column,
        ).with_sufix('target')

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

    def using_source(self, source: FeatureSourceable | BatchDataSource) -> FeatureViewStore:
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
        view_source: FeatureSourceable

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
            selected_columns = self.feature_filter
        else:
            selected_columns = set(request.needed_requests[0].all_returned_columns)

        return job.select_columns(selected_columns)

    def between_dates(self, start_date: datetime, end_date: datetime) -> RetrivalJob:
        if not isinstance(self.source, RangeFeatureSource):
            raise ValueError(
                f'The source needs to conform to RangeFeatureSource, you got a {type(self.source)}'
            )

        if self.feature_filter:
            request = self.view.request_for(self.feature_filter)
            return SelectColumnsJob(
                self.feature_filter, self.source.all_between(start_date, end_date, request)
            )

        request = self.view.request_all
        return self.source.all_between(start_date, end_date, request)

    def previous(self, days: int = 0, minutes: int = 0, seconds: int = 0) -> RetrivalJob:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days, minutes=minutes, seconds=seconds)
        return self.between_dates(start_date, end_date)

    def features_for(
        self, entities: ConvertableToRetrivalJob | RetrivalJob, event_timestamp_column: str | None = None
    ) -> RetrivalJob:
        features = {'*'}
        if self.feature_filter:
            features = self.feature_filter

        feature_refs = [f'{self.view.name}:{feature}' for feature in features]
        return self.store.features_for(
            entities,
            feature_refs,
            event_timestamp_column=event_timestamp_column,
        )

    def select(self, features: set[str]) -> FeatureViewStore:
        logger.info(f'Selecting features {features}')
        return FeatureViewStore(self.store, self.view, self.event_triggers, features)

    async def upsert(self, values: RetrivalJob | ConvertableToRetrivalJob) -> None:
        await self.store.upsert_into(FeatureLocation.feature_view(self.name), values)

    async def insert(self, values: RetrivalJob | ConvertableToRetrivalJob) -> None:
        await self.store.insert_into(FeatureLocation.feature_view(self.name), values)

    @property
    def write_input(self) -> set[str]:
        features = set()
        for request in self.view.request_all.needed_requests:
            features.update(request.all_required_feature_names)
            features.update(request.entity_names)
            if event_timestamp := request.event_timestamp:
                features.add(event_timestamp.name)
        return features

    async def write(self, values: ConvertableToRetrivalJob) -> None:
        from aligned import FileSource
        from aligned.schemas.derivied_feature import AggregateOver

        requests = self.view.request_all.needed_requests
        if self.feature_filter is not None:
            logger.info(f'Filtering features to {self.feature_filter}')
            requests = self.view.request_for(self.feature_filter).needed_requests

        if len(requests) != 1:
            raise ValueError(f'Something odd happend. Expected 1 request when writing, got {len(requests)}')

        request = requests[0]

        job = (
            RetrivalJob.from_convertable(values, request)
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
            job = job.select_columns(self.feature_filter)

        await self.batch_write(job)

    def process_input(self, values: ConvertableToRetrivalJob) -> RetrivalJob:

        request = self.view.request_all.needed_requests[0]

        job = RetrivalJob.from_convertable(values, request)

        return (
            job.fill_missing_columns().ensure_types([request]).aggregate(request).derive_features([request])
        )

    async def batch_write(self, values: ConvertableToRetrivalJob | RetrivalJob) -> None:
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
        else:
            core_job = RetrivalJob.from_convertable(values, request)

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
            await self.source.insert(job, job.retrival_requests)

    async def freshness(self) -> datetime | None:

        view = self.view
        if not view.event_timestamp:
            raise ValueError(
                f"View named '{view.name}' have no event timestamp. Therefore, unable to compute freshness"
            )

        location = FeatureLocation.feature_view(view.name)

        return (await self.source.freshness_for({location: view.event_timestamp}))[location]
