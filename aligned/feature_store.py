from __future__ import annotations
from copy import copy

import polars as pl
from aligned.lazy_imports import pandas as pd

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Union, TypeVar, Callable

from prometheus_client import Histogram

from aligned.compiler.model import ModelContractWrapper
from aligned.data_file import DataFileReference, upsert_on_column
from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    ColumnFeatureMappable,
    BatchDataSource,
)
from aligned.exceptions import UnableToFindFileException
from aligned.feature_source import (
    BatchFeatureSource,
    FeatureSource,
    FeatureSourceFactory,
    RangeFeatureSource,
    WritableFeatureSource,
)
from aligned.feature_view.feature_view import FeatureView, FeatureViewWrapper
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.retrival_job import (
    PredictionJob,
    SelectColumnsJob,
    RetrivalJob,
    StreamAggregationJob,
    SupervisedJob,
    ConvertableToRetrivalJob,
    CustomLazyPolarsJob,
)
from aligned.schemas.feature import FeatureLocation, Feature, FeatureReference
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.schemas.folder import DatasetStore
from aligned.schemas.model import EventTrigger
from aligned.schemas.model import Model as ModelSchema
from aligned.schemas.repo_definition import RepoDefinition, RepoMetadata
from aligned.sources.vector_index import VectorIndex

logger = logging.getLogger(__name__)

feature_view_write_time = Histogram(
    'feature_view_write_time',
    'The time used to write data related to a feature view',
    labelnames=['feature_view'],
)

FeatureSourceable = Union[FeatureSource, FeatureSourceFactory, None]
T = TypeVar('T')


@dataclass
class SourceRequest:
    """
    Represent a request to a source.
    This can be used validate the sources.
    """

    location: FeatureLocation
    source: CodableBatchDataSource
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
            assert splits[0]
            return (FeatureLocation(splits[1], splits[0]), splits[2])  # type: ignore
        if len(splits) == 2:
            return (FeatureLocation(splits[0], 'feature_view'), splits[1])
        else:
            raise ValueError(f'Unable to decode {splits}')


class ContractStore:

    feature_source: FeatureSource
    feature_views: dict[str, CompiledFeatureView]
    models: dict[str, ModelSchema]
    vector_indexes: dict[str, ModelSchema]

    @property
    def all_models(self) -> list[str]:
        return list(self.models.keys())

    def __init__(
        self,
        feature_views: dict[str, CompiledFeatureView],
        models: dict[str, ModelSchema],
        feature_source: FeatureSource,
        vector_indexes: dict[str, ModelSchema] | None = None,
    ) -> None:
        self.feature_source = feature_source
        self.feature_views = feature_views
        self.models = models
        self.vector_indexes = vector_indexes or {}

    @staticmethod
    def empty() -> ContractStore:
        """
        Creates a feature store with no features or models.

        ```python
        store = ContractStore.empty()

        store.add_compiled_view(MyFeatureView.compile())
        store.add_compiled_model(MyModel.compile())

        df = await store.execute_sql("SELECT * FROM my_view LIMIT 10").to_polars()
        ```
        """
        return ContractStore.from_definition(
            RepoDefinition(
                metadata=RepoMetadata(created_at=datetime.utcnow(), name='experimental'),
            )
        )

    @staticmethod
    def experimental() -> ContractStore:
        return ContractStore.empty()

    @staticmethod
    def from_definition(repo: RepoDefinition) -> ContractStore:
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
        store = ContractStore(
            feature_views={},
            models={},
            feature_source=BatchFeatureSource({}),
        )

        for view in repo.feature_views:
            store.add_compiled_view(view)

        for model in repo.models:
            store.add_compiled_model(model)

        return store

    def dummy_store(self) -> ContractStore:
        """
        Creates a new contract store that only generates dummy data
        """
        from aligned.sources.random_source import RandomDataSource

        sources: dict[str, BatchDataSource] = {}

        for view_name in self.feature_views.keys():
            sources[FeatureLocation.feature_view(view_name).identifier] = RandomDataSource()

        for model_name in self.models.keys():
            sources[FeatureLocation.model(model_name).identifier] = RandomDataSource()

        return ContractStore(
            self.feature_views, self.models, BatchFeatureSource(sources), self.vector_indexes
        )

    def source_for(self, location: FeatureLocation) -> BatchDataSource | None:
        if not isinstance(self.feature_source, BatchFeatureSource):
            return None
        return self.feature_source.sources.get(location.identifier)

    def predict_when_missing(self) -> ContractStore:
        from aligned.data_source.model_predictor import PredictModelSource

        new_store = self

        for model_name, model in self.models.items():
            if not model.exposed_model:
                continue

            new_store = new_store.update_source_for(
                FeatureLocation.model(model_name),
                PredictModelSource(
                    new_store.model(model_name),
                    cache_source=self.source_for(FeatureLocation.model(model_name)),
                ),
            )

        return new_store

    def without_model_cache(self) -> ContractStore:
        from aligned.data_source.model_predictor import PredictModelSource

        new_store = self

        for model_name, model in self.models.items():
            if not model.exposed_model:
                continue

            new_store = new_store.update_source_for(
                FeatureLocation.model(model_name), PredictModelSource(new_store.model(model_name))
            )

        return new_store

    def repo_definition(self) -> RepoDefinition:
        return RepoDefinition(
            metadata=RepoMetadata(datetime.utcnow(), name='feature_store_location.py'),
            feature_views=set(self.feature_views.values()),
            models=set(self.models.values()),
        )

    def combined_with(self, other: ContractStore) -> ContractStore:
        """
        Combines two different feature stores together.
        """
        new_store = ContractStore.empty()

        for view in self.feature_views.values():
            new_store.add_view(view)

        for view in other.feature_views.values():
            new_store.add_view(view)

        for model in self.models.values():
            new_store.add_compiled_model(model)

        for model in other.models.values():
            new_store.add_compiled_model(model)

        return new_store

    @staticmethod
    async def from_reference_at_path(
        path: str = '.', reference_file: str = 'feature_store_location.py'
    ) -> ContractStore:
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
        return ContractStore.from_definition(repo_def)

    @staticmethod
    async def from_glob(glob: str) -> ContractStore:
        """Reads and generates a feature store based on the given glob path.

        This will read the feature views, services etc in a given repo and generate a feature store.
        This can be used for fast development purposes.

        Args:
            glob (str): the files to read. E.g. `src/**/*.py`

        Returns:
            ContractStore: The generated contract store
        """
        definition = await RepoDefinition.from_glob(glob)
        return ContractStore.from_definition(definition)

    @staticmethod
    async def from_dir(path: str = '.') -> ContractStore:
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
        return ContractStore.from_definition(definition)

    def execute_sql(self, query: str) -> RetrivalJob:
        import polars as pl
        import sqlglot
        import sqlglot.expressions as exp

        expr = sqlglot.parse_one(query)
        select_expr = expr.find_all(exp.Select)

        tables = set()
        table_alias: dict[str, str] = {}
        table_columns: dict[str, set[str]] = defaultdict(set)
        unique_column_table_lookup: dict[str, str] = {}

        all_table_columns = {
            table_name: set(view.request_all.needed_requests[0].all_returned_columns)
            for table_name, view in self.feature_views.items()
        }
        all_model_columns = {
            table_name: set(model.predictions_view.request(table_name).all_returned_columns)
            for table_name, model in self.models.items()
        }

        for expr in select_expr:

            for table in expr.find_all(exp.Table):
                tables.add(table.name)
                table_alias[table.alias_or_name] = table.name

                for column in all_table_columns.get(table.name, set()).union(
                    all_model_columns.get(table.name, set())
                ):

                    if column in unique_column_table_lookup:
                        del unique_column_table_lookup[column]
                    else:
                        unique_column_table_lookup[column] = table.name

            if expr.find(exp.Star):
                for table in tables:
                    table_columns[table].update(
                        all_table_columns.get(table, set()).union(all_model_columns.get(table, set()))
                    )
            else:
                for column in expr.find_all(exp.Column):
                    source_table = table_alias.get(column.table)

                    if source_table:
                        table_columns[source_table].add(column.name)
                        continue

                    if column.table == '' and column.name in unique_column_table_lookup:
                        table_columns[unique_column_table_lookup[column.name]].add(column.name)
                        continue

                    raise ValueError(f"Unable to find table `{column.table}` for query `{query}`")

        all_features: set[str] = set()

        for table, columns in table_columns.items():
            all_features.update(f'{table}:{column}' for column in columns)

        raw_request = RawStringFeatureRequest(features=all_features)
        feature_request = self.requests_for(raw_request, None)

        request = RetrivalRequest.unsafe_combine(feature_request.needed_requests)

        async def run_query() -> pl.LazyFrame:
            dfs = {}

            for req in feature_request.needed_requests:

                if req.location.location_type == 'feature_view':
                    view = self.feature_view(req.location.name).select(req.all_feature_names).all()
                    dfs[req.location.name] = await view.to_lazy_polars()
                elif req.location.location_type == 'model':
                    model = (
                        self.model(req.location.name).all_predictions().select_columns(req.all_feature_names)
                    )
                    dfs[req.location.name] = await model.to_lazy_polars()
                else:
                    raise ValueError(f"Unsupported location: {req.location}")

            return pl.SQLContext(dfs).execute(query)

        return CustomLazyPolarsJob(
            request=request,
            method=run_query,
        )

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
        features: list[str] | list[FeatureReference],
        event_timestamp_column: str | None = None,
        model_version_as_entity: bool | None = None,
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
        assert features, 'One or more features are needed'
        raw_features = {feat.identifier if isinstance(feat, FeatureReference) else feat for feat in features}
        feature_request = RawStringFeatureRequest(features=raw_features)
        requests = self.requests_for(feature_request, event_timestamp_column, model_version_as_entity)

        feature_names = set()

        if event_timestamp_column and requests.needs_event_timestamp:
            feature_names.add(event_timestamp_column)

        for view, feature_set in feature_request.grouped_features.items():
            if feature_set != {'*'}:
                feature_names.update(feature_set)
            else:
                for request in requests.needed_requests:
                    if view.name == request.location.name:
                        feature_names.update(request.all_returned_columns)

        if not isinstance(entities, RetrivalJob):
            entities = RetrivalJob.from_convertable(entities, requests)
            feature_names.update(entities.loaded_columns)
        else:
            feature_names.update(entities.request_result.all_returned_columns)

        existing_features = set(entities.loaded_columns)

        loaded_requests = []

        for request_index in range(len(requests.needed_requests)):
            request = requests.needed_requests[request_index]
            feature_names.update(request.entity_names)

            if request.features_to_include - existing_features:
                request.features = {
                    feature for feature in request.features if feature.name not in existing_features
                }
                loaded_requests.append(request)

        if not loaded_requests:
            return entities

        new_request = FeatureRequest(requests.location, requests.features_to_include, loaded_requests)
        return self.features_for_request(new_request, entities, feature_names).inject_store(self)

    def model(self, model: str | ModelContractWrapper) -> ModelFeatureStore:
        """
        Selects a model for easy of use.

        ```python
        entities = {"trip_id": [1, 2, 3, ...]}
        preds = await store.model("my_model").predict_over(entities).to_polars()
        ```

        Returns:
            ModelFeatureStore: A new store that containes the selected model
        """
        if isinstance(model, ModelContractWrapper):
            name = model.location.name
        else:
            name = model

        return ModelFeatureStore(self.models[name], self)

    def vector_index(self, name: str | ModelContractWrapper) -> VectorIndexStore:
        if isinstance(name, ModelContractWrapper):
            name = name.location.name

        return VectorIndexStore(self, self.vector_indexes[name], index_name=name)

    def event_triggers_for(self, feature_view: str) -> set[EventTrigger]:
        triggers = self.feature_views[feature_view].event_triggers or set()
        for model in self.models.values():
            for target in model.predictions_view.classification_targets or set():
                if target.event_trigger and target.estimating.location.location_type == feature_view:
                    triggers.add(target.event_trigger)
        return triggers

    @staticmethod
    def _requests_for(
        feature_request: RawStringFeatureRequest,
        feature_views: dict[str, CompiledFeatureView],
        models: dict[str, ModelSchema],
        event_timestamp_column: str | None = None,
        model_version_as_entity: bool | None = None,
    ) -> FeatureRequest:

        features = feature_request.grouped_features
        requests: list[RetrivalRequest] = []
        entity_names = set()

        for location in feature_request.locations:
            location_name = location.name

            if location.location_type == 'model':
                model = models[location_name]
                view = model.predictions_view
                if len(features[location]) == 1 and list(features[location])[0] == '*':
                    request = view.request(location_name, model_version_as_entity or False)
                else:
                    request = view.request_for(
                        features[location], location_name, model_version_as_entity or False
                    )
                requests.append(request)
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
            elif location_name in models:
                model = models[location_name]
                feature_view = model.predictions_view

                if feature_view is None:
                    raise ValueError(f'Unable to find: {location_name}')

                if len(features[location]) == 1 and list(features[location])[0] == '*':
                    sub_request = feature_view.request(location_name, model_version_as_entity or False)
                else:
                    sub_request = feature_view.request_for(
                        features[location], location_name, model_version_as_entity or False
                    )

                requests.append(sub_request)
                entity_names.update(sub_request.entity_names)
            else:
                raise ValueError(
                    f'Unable to find: {location_name}, ' f'availible views are: {feature_views.keys()}'
                )

        if event_timestamp_column:
            entity_names.add(event_timestamp_column)
            requests = [request.with_event_timestamp_column(event_timestamp_column) for request in requests]

        else:
            requests = [request.without_event_timestamp() for request in requests]

        if not requests:
            raise ValueError(f'Unable to find any requests for: {feature_request}')

        return FeatureRequest(
            FeatureLocation.model('custom features'),
            feature_request.feature_names.union(entity_names),
            RetrivalRequest.combine(requests),
        )

    def requests_for_features(
        self, features: list[str] | list[FeatureReference], event_timestamp_column: str | None = None
    ) -> FeatureRequest:

        features = [
            feature.identifier if isinstance(feature, FeatureReference) else feature for feature in features
        ]
        return self.requests_for(RawStringFeatureRequest(set(features)), event_timestamp_column)

    def requests_for(
        self,
        feature_request: RawStringFeatureRequest,
        event_timestamp_column: str | None = None,
        model_version_as_entity: bool | None = None,
    ) -> FeatureRequest:
        return ContractStore._requests_for(
            feature_request,
            self.feature_views,
            self.models,
            event_timestamp_column=event_timestamp_column,
            model_version_as_entity=model_version_as_entity,
        )

    def feature_view(self, view: str | FeatureViewWrapper) -> FeatureViewStore:
        """
        Selects a feature view based on a name.

        From here can you query the feature view for features.

        ```python
        data = await store.feature_view('my_view').all(limit=10).to_pandas()
        ```

        Args:
            view (str): The name of the feature view

        Returns:
            FeatureViewStore: The selected feature view ready for querying
        """
        if isinstance(view, FeatureViewWrapper):
            view_name = view.location.name
        else:
            view_name = view

        feature_view = self.feature_views[view_name]
        return FeatureViewStore(self, feature_view, self.event_triggers_for(view_name))

    def add_view(self, view: CompiledFeatureView | FeatureView | FeatureViewWrapper) -> None:
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
        self.add_feature_view(view)

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
        if view.name in self.feature_views:
            raise ValueError(f'Feature view with name "{view.name}" already exists')

        if isinstance(view.source, VectorIndex):
            index_name = view.source.vector_index_name() or view.name
            self.vector_indexes[index_name] = view

        self.feature_views[view.name] = view
        if isinstance(self.feature_source, BatchFeatureSource):
            assert isinstance(self.feature_source.sources, dict)
            self.feature_source.sources[FeatureLocation.feature_view(view.name).identifier] = (
                view.materialized_source or view.source
            )

    def remove(self, location: str | FeatureLocation) -> None:
        """
        Removing a feature view or a model contract from the store.

        ```python
        store.remove("feature_view:titanic")
        # or
        location = FeatureLocation.feature_view("titanic")
        store.remove(location)
        ```

        Args:
            location (str | FeatureLocation): The contract to remove
        """
        if isinstance(location, str):
            location = FeatureLocation.from_string(location)

        if location.location_type == 'feature_view':
            del self.feature_views[location.name]
        else:
            del self.models[location.name]

        if not isinstance(self.feature_source, BatchFeatureSource):
            return
        if not isinstance(self.feature_source.sources, dict):
            return
        del self.feature_source.sources[location.identifier]

    def add(self, contract: FeatureViewWrapper | ModelContractWrapper) -> None:
        """
        Adds a feature view or a model contract

        ```python
        @feature_view(...)
        class MyFeatures:
            feature_id = String().as_entity()
            feature = Int32()

        store.add(MyFeatures)
        ```

        Args:
            contract (FeatureViewWrapper | ModelContractWrappe): The contract to add
        """
        if isinstance(contract, FeatureViewWrapper):
            self.add_feature_view(contract)
        else:
            self.add_model(contract)

    def add_feature_view(self, feature_view: FeatureView | FeatureViewWrapper | CompiledFeatureView) -> None:
        if isinstance(feature_view, FeatureViewWrapper):
            self.add_compiled_view(feature_view.compile())
        elif isinstance(feature_view, FeatureView):
            self.add_compiled_view(feature_view.compile_instance())
        else:
            self.add_compiled_view(feature_view)

    def add_model(self, model: ModelContractWrapper) -> None:
        """
        Compiles and adds the model to the store

        Args:
            model (Model): The model to add
        """
        self.add_compiled_model(model.compile())

    def add_compiled_model(self, model: ModelSchema) -> None:

        if model.name in self.models:
            raise ValueError(f'Model with name {model.name} already exists')

        self.models[model.name] = model

        source = None
        if model.predictions_view.source:
            source = model.predictions_view.source
        elif model.exposed_model:
            from aligned.data_source.model_predictor import PredictModelSource

            source = PredictModelSource(self.model(model.name))

        if isinstance(model.predictions_view.source, VectorIndex):
            index_name = model.predictions_view.source.vector_index_name() or model.name
            self.vector_indexes[index_name] = model

        if isinstance(self.feature_source, BatchFeatureSource) and source is not None:
            assert isinstance(self.feature_source.sources, dict)
            self.feature_source.sources[FeatureLocation.model(model.name).identifier] = source

    def with_source(self, source: FeatureSourceable = None) -> ContractStore:
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
        elif isinstance(source, FeatureSource):
            feature_source = source
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
        else:
            raise ValueError(
                'Setting a dedicated source needs to be either a FeatureSource, '
                f'or FeatureSourceFactory. Got: {type(source)}'
            )

        return ContractStore(
            feature_views=self.feature_views,
            models=self.models,
            feature_source=feature_source,
        )

    def sources_of_type(self, source_type: type[T], function: Callable[[T, FeatureLocation], None]) -> None:

        if not isinstance(self.feature_source, BatchFeatureSource):
            raise ValueError(
                f'.update_source_for(...) needs a `BatchFeatureSource`, got {type(self.feature_source)}'
            )

        assert isinstance(self.feature_source.sources, dict), 'Can only operate on a dict'

        for location, source in self.feature_source.sources.items():
            if not isinstance(source, source_type):
                continue

            loc = FeatureLocation.from_string(location)
            function(source, loc)

    def update_source_for(self, location: FeatureLocation | str, source: BatchDataSource) -> ContractStore:
        if not isinstance(self.feature_source, BatchFeatureSource):
            raise ValueError(
                f'.update_source_for(...) needs a `BatchFeatureSource`, got {type(self.feature_source)}'
            )

        if isinstance(location, str):
            location = FeatureLocation.from_string(location)

        new_source = BatchFeatureSource(copy(self.feature_source.sources))
        assert isinstance(new_source.sources, dict)
        new_source.sources[location.identifier] = source

        return ContractStore(
            feature_views=self.feature_views,
            models=self.models,
            feature_source=new_source,
        )

    def offline_store(self) -> ContractStore:
        """
        Will set the source to the defined batch sources.

        Returns:
            FeatureStore: A new feature store that loads features from the batch sources
        """
        return self.with_source()

    def use_application_sources(self) -> ContractStore:
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
        return ContractStore(
            feature_views=self.feature_views,
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

    def write_request_for(self, location: FeatureLocation) -> RetrivalRequest:

        if location.location_type == 'feature_view':
            return self.feature_views[location.name].request_all.needed_requests[0]
        elif location.location_type == 'model':
            return self.models[location.name].predictions_view.request('write', model_version_as_entity=True)
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
            await source.insert(values, write_request)
        elif isinstance(source, DataFileReference):
            import polars as pl

            columns = write_request.all_returned_columns
            new_df = (await values.to_lazy_polars()).select(columns)

            try:
                existing_df = await source.to_lazy_polars()
                write_df = (
                    pl.concat([new_df, existing_df.select(columns)], how='vertical_relaxed').collect().lazy()
                )
            except (UnableToFindFileException, pl.ComputeError):
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
            await source.upsert(values, write_request)
        elif isinstance(source, DataFileReference):
            new_df = (await values.to_lazy_polars()).select(write_request.all_returned_columns)
            entities = list(write_request.entity_names)
            try:
                existing_df = await source.to_lazy_polars()
                write_df = upsert_on_column(entities, new_df, existing_df)
            except (UnableToFindFileException, pl.ComputeError):
                write_df = new_df

            await source.write_polars(write_df)
        else:
            raise ValueError(f'The source {type(source)} do not support writes')

    async def overwrite(
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
            await source.overwrite(values, write_request)
        elif isinstance(source, DataFileReference):
            df = (await values.to_lazy_polars()).select(write_request.all_returned_columns)
            await source.write_polars(df)
        else:
            raise ValueError(f'The source {type(source)} do not support writes')


FeatureStore = ContractStore


@dataclass
class ModelFeatureStore:

    model: ModelSchema
    store: ContractStore
    selected_version: str | None = None

    @property
    def location(self) -> FeatureLocation:
        return FeatureLocation.model(self.model.name)

    @property
    def dataset_store(self) -> DatasetStore | None:
        return self.model.dataset_store

    def has_one_source_for_input_features(self) -> bool:
        """
        If the input features are from the same source.

        This can be interesting to know in order to automatically predict over
        the input.
        E.g. predict over all data in the source.
        """
        version = self.selected_version or self.model.features.default_version
        features = self.model.features.features_for(version)
        locations = {feature.location for feature in features}
        return len(locations) == 1

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
        return self.input_request(except_features, event_timestamp_column)

    def input_request(
        self, except_features: set[str] | None = None, event_timestamp_column: str | None = None
    ) -> FeatureRequest:
        feature_refs = self.raw_string_features(except_features or set())
        if not feature_refs:
            raise ValueError(f"No features to request for model '{self.model.name}'")

        return self.store.requests_for(
            RawStringFeatureRequest(feature_refs),
            event_timestamp_column,
        )

    def prediction_request(
        self, exclude_features: set[str] | None = None, model_version_as_entity: bool = False
    ) -> RetrivalRequest:
        if not self.model.predictions_view:
            raise ValueError(f'Model {self.model.name} has no predictions view')

        if exclude_features is None:
            exclude_features = set()

        features = {
            feature.name
            for feature in self.model.predictions_view.full_schema
            if feature.name not in exclude_features
        }

        return self.model.predictions_view.request_for(
            features, self.model.name, model_version_as_entity=model_version_as_entity
        )

    def needed_entities(self) -> set[Feature]:
        return self.request().request_result.entities

    def feature_references_for(self, version: str) -> list[FeatureReference]:
        return self.model.features.features_for(version)

    def has_exposed_model(self) -> bool:
        return self.model.exposed_model is not None

    def predict_over(
        self,
        entities: ConvertableToRetrivalJob | RetrivalJob,
    ) -> PredictionJob:
        predictor = self.model.exposed_model
        if not predictor:
            raise ValueError(
                f'Model {self.model.name} has no predictor set. '
                'This can be done by setting the `exposed_at` value'
            )

        returned_request = self.request().needed_requests

        if not isinstance(entities, RetrivalJob):
            entities = RetrivalJob.from_convertable(entities, returned_request)

        return PredictionJob(entities, self.model, self.store, returned_request)

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

        job = None

        if isinstance(entities, (dict, pl.DataFrame, pd.DataFrame)):

            existing_keys = set()
            if isinstance(entities, dict):
                existing_keys = set(entities.keys())
            elif isinstance(entities, (pl.DataFrame, pd.DataFrame)):
                existing_keys = set(entities.columns)

            subset_request = self.request(existing_keys, event_timestamp_column)

            needs_core_features = False

            for req in subset_request.needed_requests:
                missing_keys = set(req.feature_names) - existing_keys
                if missing_keys:
                    needs_core_features = True
                    break

            if not needs_core_features:
                job = (
                    RetrivalJob.from_convertable(entities, request)
                    .derive_features(request.needed_requests)
                    .inject_store(self.store)
                )

        if job is None:
            job = self.store.features_for(
                entities, list(features), event_timestamp_column=event_timestamp_column
            )

        return job

    async def freshness(self) -> dict[FeatureLocation, datetime | None]:
        return await self.input_freshness()

    async def input_freshness(self) -> dict[FeatureLocation, datetime | None]:
        locs: dict[FeatureLocation, Feature] = {}

        other_locs: set[FeatureLocation] = set()

        for req in self.request().needed_requests:
            if req.event_timestamp:
                locs[req.location] = req.event_timestamp.as_feature()

            for feature in req.derived_features:
                if feature.loads_feature:
                    other_locs.add(feature.loads_feature.location)

        if self.model.exposed_model:
            other_locs.update(await self.model.exposed_model.depends_on())

        for loc in other_locs:
            if loc in locs:
                continue

            if loc.location_type == 'model':
                event_timestamp = (
                    self.store.model(loc.name).model.predictions_view.as_view(loc.name).freshness_feature
                )
            else:
                event_timestamp = self.store.feature_view(loc.name).view.freshness_feature

            if event_timestamp:
                locs[loc] = event_timestamp

        return await self.store.feature_source.freshness_for(locs)

    async def prediction_freshness(self) -> datetime | None:
        feature = (
            self.store.model(self.model.name)
            .model.predictions_view.as_view(self.model.name)
            .freshness_feature
        )
        if not feature:
            return None
        freshness = await self.store.feature_source.freshness_for({self.location: feature})
        return freshness[self.location]

    def with_labels(self, label_refs: set[FeatureReference] | None = None) -> SupervisedModelFeatureStore:
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
        self,
        entities: ConvertableToRetrivalJob | RetrivalJob,
        event_timestamp_column: str | None = None,
        model_version_as_entity: bool | None = None,
    ) -> RetrivalJob:

        location_id = self.location.identifier
        return self.store.features_for(
            entities,
            features=[f'{location_id}:*'],
            event_timestamp_column=event_timestamp_column,
            model_version_as_entity=model_version_as_entity,
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

        return (
            source.all_data(request, limit=limit)
            .inject_store(self.store)
            .select_columns(set(request.all_returned_columns))
        )

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
        locs = {req.location for req in self.request().needed_requests}
        label_refs = self.model.predictions_view.labels_estimates_refs()
        if label_refs:
            for ref in label_refs:
                locs.add(ref.location)
        return locs

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
    store: ContractStore
    labels_estimates_refs: set[FeatureReference]

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

    store: ContractStore
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

    def select_columns(self, columns: list[str], limit: int | None = None) -> RetrivalJob:
        return self.select(set(columns)).all(limit)

    def all(self, limit: int | None = None) -> RetrivalJob:
        return self.all_columns(limit)

    def filter(self, filter: pl.Expr | str) -> RetrivalJob:
        return self.all().filter(filter)

    def all_columns(self, limit: int | None = None) -> RetrivalJob:
        if not isinstance(self.source, RangeFeatureSource):
            raise ValueError(f'The source ({self.source}) needs to conform to RangeFeatureSource')

        request = self.view.request_all
        if self.feature_filter:
            request = self.view.request_for(self.feature_filter)

        job = (
            self.source.all_for(request, limit)
            .ensure_types(request.needed_requests)
            .derive_features(request.needed_requests)
            .inject_store(self.store)
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
                list(self.feature_filter), self.source.all_between(start_date, end_date, request)
            )

        request = self.view.request_all
        return self.source.all_between(start_date, end_date, request).inject_store(self.store)

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
            await self.source.insert(job, job.retrival_requests[0])

    async def freshness(self) -> datetime | None:

        view = self.view
        if not view.event_timestamp:
            raise ValueError(
                f"View named '{view.name}' have no event timestamp. Therefore, unable to compute freshness"
            )

        location = FeatureLocation.feature_view(view.name)

        return (await self.source.freshness_for({location: view.event_timestamp.as_feature()}))[location]


class VectorIndexStore:

    store: ContractStore
    model: ModelSchema
    index_name: str

    def __init__(self, store: ContractStore, model: ModelSchema, index_name: str):
        if model.predictions_view.source is None:
            raise ValueError(f"An output source on the model {model.name} is needed")

        if not isinstance(model.predictions_view.source, VectorIndex):
            message = (
                f"An output source on the model {model.name} needs to be of type VectorIndex,"
                f"got {type(model.predictions_view.source)}"
            )
            raise ValueError(message)

        self.store = store
        self.model = model
        self.index_name = index_name

    def nearest_n_to(
        self, entities: RetrivalJob | ConvertableToRetrivalJob, number_of_records: int
    ) -> RetrivalJob:
        source = self.model.predictions_view.source
        assert isinstance(source, VectorIndex)

        embeddings = self.model.predictions_view.embeddings()
        n_embeddings = len(embeddings)

        if n_embeddings == 0:
            raise ValueError(f"Need at least one embedding to search. Got {n_embeddings}")
        if n_embeddings > 1:
            raise ValueError('Got more than one embedding, it is therefore unclear which to use.')

        embedding = embeddings[0]
        response = self.model.predictions_view.request(self.model.name)

        def contains_embedding() -> bool:
            if isinstance(entities, RetrivalJob):
                return embedding.name in entities.loaded_columns
            elif isinstance(entities, dict):
                return embedding.name in entities
            elif isinstance(entities, (pl.DataFrame, pd.DataFrame, pl.LazyFrame)):
                return embedding.name in entities.columns
            raise ValueError('Unable to determine if the entities contains the embedding')

        if self.model.exposed_model and not contains_embedding():
            model_store = self.store.model(self.model.name)
            features: RetrivalJob = model_store.predict_over(entities)
        else:
            # Assumes that we can lookup the embeddings from the source
            feature_ref = FeatureReference(
                embedding.name, FeatureLocation.model(self.model.name), dtype=embedding.dtype
            )
            features: RetrivalJob = self.store.features_for(entities, features=[feature_ref.identifier])

        return source.nearest_n_to(features, number_of_records, response)

    def as_langchain_retriver(self, number_of_docs: int = 5):
        from aligned.exposed_model.langchain import AlignedRetriver

        return AlignedRetriver(store=self.store, index_name=self.index_name, number_of_docs=number_of_docs)
