from __future__ import annotations
from copy import copy

from aligned.compiler.feature_factory import FeatureFactory, FeatureReferencable
import polars as pl
from aligned.config_value import ConfigValue

from aligned.lazy_imports import pandas as pd, _PANDAS_AVAILABLE

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Iterable,
    Protocol,
    Sequence,
    Union,
    TypeVar,
    Callable,
    TYPE_CHECKING,
    overload,
)

from aligned.compiler.model import ModelContractWrapper
from aligned.data_file import DataFileReference, upsert_on_column
from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    ColumnFeatureMappable,
    BatchDataSource,
)
from aligned.exceptions import ConfigurationError, UnableToFindFileException
from aligned.feature_source import (
    FeatureSource,
    FeatureSourceFactory,
    WritableFeatureSource,
)
from aligned.feature_view.feature_view import (
    BatchSourceable,
    FeatureView,
    FeatureViewWrapper,
    resolve_source,
)
from aligned.request.retrieval_request import FeatureRequest, RetrievalRequest
from aligned.retrieval_job import (
    FilterRepresentable,
    SelectColumnsJob,
    RetrievalJob,
    StreamAggregationJob,
    ConvertableToRetrievalJob,
    CustomLazyPolarsJob,
)
from aligned.schemas.feature import (
    FeatureLocation,
    Feature,
    FeatureReference,
    ConvertableToLocation,
    convert_to_location,
)
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.schemas.model import EventTrigger
from aligned.schemas.model import Model as ModelSchema
from aligned.schemas.repo_definition import RepoDefinition, RepoMetadata
from aligned.schemas.transformation import Expression
from aligned.sources.in_mem_source import InMemorySource, RetrievalJobSource
from aligned.sources.local import Deletable, StorageFileReference
from aligned.sources.vector_index import VectorIndex
from aligned.model_store import ModelFeatureStore
from aligned.validation.interface import PolarsValidator, Validator

if TYPE_CHECKING:
    from aligned.sources.random_source import FillMode

logger = logging.getLogger(__name__)


FeatureSourceable = Union[FeatureSource, FeatureSourceFactory, None]
T = TypeVar("T")


class DataStore(Protocol):
    def features_for(
        self,
        entities: ConvertableToRetrievalJob | RetrievalJob,
        features: list[str] | list[FeatureReference] | list[FeatureReferencable],
        event_timestamp_column: str | None = None,
        model_version_as_entity: bool | None = None,
    ) -> RetrievalJob: ...

    def filter(self, filter: FilterRepresentable) -> RetrievalJob: ...

    def all(self, limit: int | None = None) -> RetrievalJob: ...


@dataclass
class SourceRequest:
    """
    Represent a request to a source.
    This can be used validate the sources.
    """

    location: FeatureLocation
    source: CodableBatchDataSource
    request: RetrievalRequest


@dataclass
class RawStringFeatureRequest:
    features: set[str]

    @property
    def locations(self) -> set[FeatureLocation]:
        return {
            RawStringFeatureRequest.unpack_feature(feature)[0]
            for feature in self.features
        }

    @property
    def grouped_features(self) -> dict[FeatureLocation, set[str]]:
        unpacked_features = [
            RawStringFeatureRequest.unpack_feature(feature) for feature in self.features
        ]
        grouped = defaultdict(set)
        for feature_view, feature in unpacked_features:
            grouped[feature_view].add(feature)
        return grouped

    @property
    def feature_names(self) -> set[str]:
        return {
            RawStringFeatureRequest.unpack_feature(feature)[1]
            for feature in self.features
        }

    @staticmethod
    def unpack_feature(feature: str) -> tuple[FeatureLocation, str]:
        splits = feature.split(":")
        if len(splits) == 3:
            assert splits[0]
            return (FeatureLocation(splits[1], splits[0]), splits[2])  # type: ignore
        if len(splits) == 2:
            return (FeatureLocation(splits[0], "feature_view"), splits[1])
        else:
            raise ValueError(f"Unable to decode {splits}")


class ContractStore:
    feature_views: dict[str, CompiledFeatureView]
    models: dict[str, ModelSchema]
    vector_indexes: dict[str, ModelSchema]
    sources: dict[FeatureLocation, BatchDataSource]

    @property
    def all_models(self) -> list[str]:
        return list(self.models.keys())

    @property
    def source_types(self) -> dict[str, type[BatchDataSource]]:
        return {
            source.job_group_key(): type(source) for source in self.sources.values()
        }

    def __init__(
        self,
        feature_views: dict[str, CompiledFeatureView],
        models: dict[str, ModelSchema],
        vector_indexes: dict[str, ModelSchema] | None = None,
        sources: dict[FeatureLocation, BatchDataSource] | None = None,
    ) -> None:
        self.vector_indexes = vector_indexes or {}
        self.feature_views = feature_views
        self.models = models
        if sources is not None:
            self.sources = sources
        else:
            self.sources = {}
            for view in feature_views.values():
                self.sources[FeatureLocation.feature_view(view.name)] = (
                    view.materialized_source or view.source
                )
            for model in models.values():
                if model.predictions_view.source:
                    self.sources[FeatureLocation.model(model.name)] = (
                        model.predictions_view.source
                    )

    @staticmethod
    def empty() -> ContractStore:
        """
        Creates a feature store with no features or models.

        Examples:
            ```python
            store = ContractStore.empty()

            store.add(MyFeatureView)
            store.add(MyModel)

            df = await store.execute_sql("SELECT * FROM my_view LIMIT 10").to_polars()
            ```
        Returns:
            ContractStore: An empty store with new views or models
        """
        return ContractStore.from_definition(
            RepoDefinition(
                metadata=RepoMetadata(
                    created_at=datetime.utcnow(), name="experimental"
                ),
            )
        )

    @staticmethod
    def with_contracts(
        *contracts: FeatureViewWrapper | ModelContractWrapper,
    ) -> ContractStore:
        store = ContractStore.empty()
        for contract in contracts:
            store.add(contract)
        return store

    @staticmethod
    def experimental() -> ContractStore:
        return ContractStore.empty()

    @staticmethod
    def from_contracts(
        contracts: list[FeatureViewWrapper | ModelContractWrapper],
    ) -> ContractStore:
        store = ContractStore.empty()
        for contract in contracts:
            store.add(contract)
        return store

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

        Returns:
            FeatureStore: A ready to use feature store
        """
        store = ContractStore(
            feature_views={},
            models={},
        )

        for view in repo.feature_views:
            store.add_compiled_view(view)

        for model in repo.models:
            store.add_compiled_model(model)

        return store

    async def write_to(self, file: StorageFileReference) -> None:
        repo_def = self.repo_definition()

        data = repo_def.to_json(omit_none=True)
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = data

        await file.write(data_bytes)  # type: ignore

    async def freshness_for(self, loc: FeatureLocation) -> datetime | None:
        if loc.location_type == "feature_view":
            view = self.feature_views[loc.identifier]
            source = self.source_for(loc)

            if source and view.freshness_feature:
                return await source.freshness(view.freshness_feature)
            else:
                return None
        elif loc.location_type == "model":
            model = self.models[loc.identifier]
            source = self.source_for(loc)

            if not source:
                return None

            feat = model.predictions_view.freshness_feature
            if not feat:
                return None

            return await source.freshness(feat)

        return None

    def dummy_store(self, fill_mode: FillMode = "duplicate") -> ContractStore:
        """
        Creates a new contract store that only generates dummy data

        Args:
            fill_mode (FillMode): The type of random fill to do for missing values. Defaults to generating one random sample, and duplicating that for all rows.

        Returns:
            ContractStore: A new store that only contains random sources
        """
        from aligned.sources.random_source import RandomDataSource

        sources: dict[FeatureLocation, BatchDataSource] = {}

        for view_name in self.feature_views.keys():
            sources[FeatureLocation.feature_view(view_name)] = RandomDataSource(
                fill_mode=fill_mode
            )

        for model_name in self.models.keys():
            sources[FeatureLocation.model(model_name)] = RandomDataSource(
                fill_mode=fill_mode
            )

        return ContractStore(
            self.feature_views,
            self.models,
            sources=sources,
            vector_indexes=self.vector_indexes,
        )

    def source_for(self, location: ConvertableToLocation) -> BatchDataSource | None:
        location = convert_to_location(location)
        return self.sources.get(location)

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
                FeatureLocation.model(model_name),
                PredictModelSource(new_store.model(model_name)),
            )

        return new_store

    def repo_definition(self) -> RepoDefinition:
        return RepoDefinition(
            metadata=RepoMetadata(datetime.utcnow(), name="feature_store_location.py"),
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
        path: str = ".", reference_file: str = "feature_store_location.py"
    ) -> ContractStore:
        """Looks for a file reference struct, and loads the associated repo.

        This can be used for changing which feature store definitions
        to read based on defined environment variables.

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
    async def from_dir(
        path: str = ".", exclude_glob: list[str] | None = None
    ) -> ContractStore:
        """Reads and generates a feature store based on the given directory's content.

        This will read the feature views, services etc in a given repo and generate a feature store.
        This can be used for fast development purposes.

        If you rather want a more flexible deployable solution.
        Consider using `FeatureStore.from_reference_at_path(...)` which will can read an existing
        generated file from different storages, based on an environment variable.

        Args:
            path (str, optional): the directory to read from. Defaults to ".".
            exclude_glob (list[str], optional): the globs to ignore

        Returns:
            FeatureStore: The generated feature store
        """
        definition = await RepoDefinition.from_path(path, exclude_glob)
        return ContractStore.from_definition(definition)

    def execute_sql(self, query: str) -> RetrievalJob:
        from aligned.sql import request_for_sql, glot_to_polars, extract_raw_values

        requests = request_for_sql(query, self)

        if not requests:
            raw_frame = extract_raw_values(query)
            assert raw_frame is not None, f"Unable to extract raw values from {query}"
            return RetrievalJob.from_polars_df(raw_frame, [])

        async def run_query() -> pl.LazyFrame:
            dfs = {}

            for req, filter in requests:
                if req.location.location_type == "feature_view":
                    job = (
                        self.feature_view(req.location.name)
                        .select(req.all_feature_names)
                        .all()
                    )
                elif req.location.location_type == "model":
                    job = (
                        self.model(req.location.name)
                        .all_predictions()
                        .select_columns(req.all_feature_names)
                    )
                else:
                    raise ValueError(f"Unsupported location: {req.location}")

                if filter is not None:
                    filter_exp = glot_to_polars(filter.this)
                    logger.info(f"Adding filter to '{req.location.name}' - '{filter}'")
                    job = job.filter(filter_exp)

                dfs[req.location.name] = await job.to_lazy_polars()

            logger.info("Executing SQL")
            return pl.SQLContext(dfs).execute(query)

        return CustomLazyPolarsJob(
            request=RetrievalRequest.unsafe_combine([req for req, _ in requests]),
            method=run_query,
        )

    def features_for_request(
        self,
        request: FeatureRequest,
        entities: ConvertableToRetrievalJob | RetrievalJob,
        feature_names: set[str],
    ) -> RetrievalJob:
        if isinstance(entities, RetrievalJob):
            facts = entities
        else:
            facts = RetrievalJob.from_convertable(entities, request)

        from aligned.retrieval_job import CombineFactualJob

        core_requests = [
            (self.sources[request.location], request)
            for request in request.needed_requests
            if request.location in self.sources
        ]
        source_groupes = {
            self.sources[request.location].job_group_key()
            for request in request.needed_requests
            if request.location in self.sources
        }

        logger.debug(f"Loading the sources in the following groups: {source_groupes}")
        loaded_columns = set(facts.loaded_columns)

        def needs_to_load_source(requests: list[RetrievalRequest]) -> bool:
            for req in requests:
                if set(req.feature_names) - loaded_columns:
                    return True

                for feat in req.derived_features:
                    if (
                        set(
                            depends_on.name
                            for depends_on in feat.depending_on
                            if depends_on.location != req.location
                        )
                        - loaded_columns
                    ):
                        return True

                for feat in req.aggregated_features:
                    if set(feat.depending_on_names) - loaded_columns:
                        return True
            return False

        # The combined views basically, as they have no direct
        combined_requests = [
            request
            for request in request.needed_requests
            if request.location.identifier not in self.sources
        ]

        jobs: list[RetrievalJob] = []
        for source_group in source_groupes:
            requests_with_source = [
                (source, req)
                for source, req in core_requests
                if source.job_group_key() == source_group
            ]
            requests = [req for _, req in requests_with_source]

            if needs_to_load_source(requests):
                logger.debug(
                    f"Loading features from source with group name '{source_group}'"
                )
                job = (
                    self.source_types[source_group]
                    .multi_source_features_for(
                        facts=facts, requests=requests_with_source
                    )
                    .ensure_types(requests)
                    .derive_features()
                )
            else:
                job = facts.derive_features(requests)

            if len(requests) == 1 and requests_with_source[0][1].aggregated_features:
                req = requests_with_source[0][1]
                job = job.aggregate(req)

            jobs.append(job)

        fact_features = loaded_columns - set(request.request_result.entity_columns)
        if fact_features:
            jobs.append(facts)

        assert len(jobs) != 0, "Should at least contain one job. Something is wrong"

        if len(combined_requests) > 0 or len(jobs) > 1:
            return CombineFactualJob(
                jobs=jobs,
                combined_requests=combined_requests,
            ).derive_features()
        else:
            return jobs[0].derive_features()

    def features_for(
        self,
        entities: ConvertableToRetrievalJob | RetrievalJob,
        features: Sequence[
            str
            | FeatureReference
            | FeatureReferencable
            | FeatureViewWrapper
            | ModelContractWrapper
        ],
        event_timestamp_column: str | None = None,
        model_version_as_entity: bool | None = None,
    ) -> RetrievalJob:
        """
        Returns a set of features given a set of entities.

        ```python
        entities = { "user_id": [1, 2, 3, ...] }
        job = store.features_for(entities, features=["user:time_since_last_login", ...])
        data = await job.to_pandas()
        ```

        Args:
            entities (dict[str, list] | RetrievalJob): The entities to load data for
            features (list[str]): A list of features to load. Use the format (<feature_view>:<feature>)

        Returns:
            RetrievalJob: A job that knows how to fetch the features
        """
        assert features, "One or more features are needed"

        raw_features: set[str] = set()

        for feature in features:
            if isinstance(feature, str):
                raw_features.add(feature)
            elif isinstance(feature, FeatureReference):
                raw_features.add(feature.identifier)
            elif isinstance(feature, FeatureReferencable):
                raw_features.add(feature.feature_reference().identifier)
            elif isinstance(feature, FeatureViewWrapper):
                raw_features.update(
                    feat.as_reference(feature.location).identifier
                    for feat in feature.compile().request_all.request_result.features
                )
            elif isinstance(feature, ModelContractWrapper):
                raw_features.update(
                    feat.as_reference(feature.location).identifier
                    for feat in feature.compile().request_all_predictions.request_result.features
                )
            else:
                raise ValueError(
                    f"Unable to look up feature of type {type(feature)} - {feature}"
                )

        feature_request = RawStringFeatureRequest(features=raw_features)
        requests = self.requests_for(
            feature_request, event_timestamp_column, model_version_as_entity
        )
        logger.debug(
            f"Preparing job to fetch data from '{[req.location for req in requests.needed_requests]}'"
        )

        feature_names = set()

        if event_timestamp_column and requests.needs_event_timestamp:
            feature_names.add(event_timestamp_column)

        for view, feature_set in feature_request.grouped_features.items():
            if feature_set != {"*"}:
                feature_names.update(feature_set)
            else:
                for request in requests.needed_requests:
                    if view.name == request.location.name:
                        feature_names.update(request.all_returned_columns)

        if not isinstance(entities, RetrievalJob):
            logger.debug("Converting entities into a RetrievalJob")
            if isinstance(entities, pl.DataFrame) and entities.is_empty():
                return RetrievalJob.from_convertable(
                    entities.with_columns(
                        *[
                            pl.lit(None).alias(feat.name).cast(feat.dtype.polars_type)
                            for feat in requests.request_result.features
                            if feat.name not in entities.columns
                        ]
                    ),
                    requests,
                )
            if (
                _PANDAS_AVAILABLE
                and isinstance(entities, pd.DataFrame)
                and entities.empty
            ):
                return RetrievalJob.from_convertable(entities, requests)
            if isinstance(entities, list) and not entities:
                return RetrievalJob.from_convertable(entities, requests)

            entities = RetrievalJob.from_convertable(entities, requests)
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
                    feature
                    for feature in request.features
                    if feature.name not in existing_features
                }
                loaded_requests.append(request)
            else:
                logger.debug(
                    f"Dropping request from {request.location} as all features where already loaded"
                )

        if not loaded_requests:
            from aligned.local.job import LiteralRetrievalJob

            if isinstance(entities, LiteralRetrievalJob) and not entities.requests:
                entities.requests = requests.needed_requests
            return entities

        new_request = FeatureRequest(
            requests.location, requests.features_to_include, loaded_requests
        )
        logger.debug(
            f"Updated request for following locs: '{[req.location for req in new_request.needed_requests]}'"
        )
        return self.features_for_request(
            new_request, entities, feature_names
        ).inject_store(self)

    def needed_configs_for(self, location: ConvertableToLocation) -> list[ConfigValue]:
        location = convert_to_location(location)

        if location.location_type == "feature_view":
            source = self.source_for(location)
            if source:
                return source.needed_configs()
            else:
                return []
        elif location.location_type == "model":
            model_store = self.model(location.name)
            model = model_store.model

            model_source = self.source_for(location)
            configs = model_source.needed_configs() if model_source else []

            if model.exposed_model:
                configs.extend(model.exposed_model.needed_configs())

            for loc in model_store.depends_on():
                source = self.source_for(loc)
                configs.extend(model_source.needed_configs() if model_source else [])
            return configs
        return []

    def raise_on_missing_config_for(self, location: ConvertableToLocation) -> None:
        location = convert_to_location(location)

        missing_configs = []
        for config in self.needed_configs_for(location):
            try:
                _ = config.read()
            except ValueError as error:
                missing_configs.append(error)

        if missing_configs:
            missing_config_error = "- " + "\n- ".join(
                [str(config) for config in missing_configs]
            )
            raise ConfigurationError(missing_config_error)

    def model(self, model: str | ModelContractWrapper) -> ModelFeatureStore:
        """
        Selects a model for easy of use.

        ```python
        entities = {"trip_id": [1, 2, 3, ...]}
        preds = await store.model("my_model").predict_over(entities).to_polars()
        ```

        Returns:
            ModelFeatureStore: A new store that contains the selected model
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
                if (
                    target.event_trigger
                    and target.estimating.location.location_type == feature_view
                ):
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
        requests: list[RetrievalRequest] = []
        entity_names = set()

        for location in feature_request.locations:
            location_name = location.name

            if location.location_type == "model":
                model = models[location_name]
                view = model.predictions_view
                if len(features[location]) == 1 and list(features[location])[0] == "*":
                    request = view.request(
                        location_name, model_version_as_entity or False
                    )
                else:
                    request = view.request_for(
                        features[location],
                        location_name,
                        model_version_as_entity or False,
                    )
                requests.append(request)
                entity_names.update(request.entity_names)

            elif location_name in feature_views:
                feature_view = feature_views[location_name]

                if len(features[location]) == 1 and list(features[location])[0] == "*":
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
                    raise ValueError(f"Unable to find: {location_name}")

                if len(features[location]) == 1 and list(features[location])[0] == "*":
                    sub_request = feature_view.request(
                        location_name, model_version_as_entity or False
                    )
                else:
                    sub_request = feature_view.request_for(
                        features[location],
                        location_name,
                        model_version_as_entity or False,
                    )

                requests.append(sub_request)
                entity_names.update(sub_request.entity_names)
            else:
                raise ValueError(
                    f"Unable to find: {location_name}, "
                    f"available views are: {feature_views.keys()}"
                )

        if event_timestamp_column:
            entity_names.add(event_timestamp_column)
            requests = [
                request.with_event_timestamp_column(event_timestamp_column)
                for request in requests
            ]

        if not requests:
            raise ValueError(f"Unable to find any requests for: {feature_request}")

        return FeatureRequest(
            FeatureLocation.model("custom features"),
            feature_request.feature_names.union(entity_names),
            RetrievalRequest.combine(requests),
        )

    def requests_for_features(
        self,
        features: Iterable[str] | list[FeatureReference],
        event_timestamp_column: str | None = None,
    ) -> FeatureRequest:
        features = [
            feature.identifier if isinstance(feature, FeatureReference) else feature
            for feature in features
        ]
        return self.requests_for(
            RawStringFeatureRequest(set(features)), event_timestamp_column
        )

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

    @overload
    def contract(self, view: FeatureViewWrapper) -> FeatureViewStore: ...

    @overload
    def contract(self, view: ModelContractWrapper) -> ModelFeatureStore: ...

    @overload
    def contract(self, view: str) -> DataStore: ...

    def contract(
        self, view: str | FeatureViewWrapper | ModelContractWrapper
    ) -> FeatureViewStore | ModelFeatureStore | DataStore:
        """
        Selects a contract based on a name or wrapper.

        From here can you query the feature view for features.

        ```python
        @data_contract(...)
        class SomeData:
            ...

        data = await store.contract(SomeData).all().to_polars()

        # Or by name
        data = await store.contract('some_data').all().to_polars()


        @model_contract(...)
        class SomeModel:
            ...

        data = await store.contract(SomeModel).all_predictions().to_polars()
        ```

        Args:
            view (str): The name of the feature view

        Returns:
            FeatureViewStore: The selected feature view ready for querying
        """
        if isinstance(view, FeatureViewWrapper):
            return self.feature_view(view)
        elif isinstance(view, ModelContractWrapper):
            return self.model(view)
        else:
            if view in self.feature_views:
                return self.feature_view(view)
            else:
                return self.model(view)

    def data_contract(self, contract: str | FeatureViewWrapper) -> FeatureViewStore:
        """
        Selects a data contract based on a name or contract wrapper.

        From here can you query the data contract for features.

        ```python
        data = await store.feature_view('my_view').all(limit=10).to_polars()
        ```

        Args:
            contract (str | FeatureViewWrapper): The name of the data contract

        Returns:
            FeatureViewStore: The selected data contract ready for querying
        """
        return self.feature_view(contract)

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

    def add_view(
        self, view: CompiledFeatureView | FeatureView | FeatureViewWrapper
    ) -> None:
        """
        Compiles and adds the feature view to the store

        Examples:
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

        Examples:
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
            self.vector_indexes[index_name] = view  # type: ignore

        self.feature_views[view.name] = view
        self.sources[FeatureLocation.feature_view(view.name)] = (
            view.materialized_source or view.source
        )

    def remove(self, location: ConvertableToLocation) -> None:
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
        location = convert_to_location(location)

        if location.location_type == "feature_view":
            del self.feature_views[location.name]
        else:
            del self.models[location.name]

        del self.sources[location]

    def add(self, contract: FeatureViewWrapper | ModelContractWrapper) -> None:
        """
        Adds a feature view or a model contract

        Args:
            contract (FeatureViewWrapper | ModelContractWrappe): The contract to add

        Examples:
            ```python
            @feature_view(...)
            class MyFeatures:
                feature_id = String().as_entity()
                feature = Int32()

            store.add(MyFeatures)
            ```
        """
        if isinstance(contract, FeatureViewWrapper):
            self.add_feature_view(contract)
        else:
            self.add_model(contract)

    def add_feature_view(
        self, feature_view: FeatureView | FeatureViewWrapper | CompiledFeatureView
    ) -> None:
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
            raise ValueError(f"Model with name {model.name} already exists")

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

        if source:
            self.sources[FeatureLocation.model(model.name)] = source

    @overload
    def sources_of_type(
        self, source_type: type[T], function: None = None
    ) -> list[tuple[T, FeatureLocation]]: ...

    @overload
    def sources_of_type(
        self, source_type: type[T], function: Callable[[T, FeatureLocation], None]
    ) -> None: ...

    def sources_of_type(
        self,
        source_type: type[T],
        function: Callable[[T, FeatureLocation], None] | None = None,
    ) -> None | list[tuple[T, FeatureLocation]]:
        """
        Process all sources of a specific type

        ```python
        store = await ContractStore.from_dir()

        sources = store.sources_of_type(DatabricksSource)
        print(sources)

        # Or if you want to operate over the sources
        def update_databricks_source(source: DatabricksSource, loc: FeatureLocation) -> None:
            source.config = DatabricksConfig.serverless()

        store.sources_of_type(
            DatabricksSource,
            update_databricks_source
        )
        ```

        Args:
            source_type: The source type you want to process
            function (Callable[[T, FeatureLocation], None]): The function that process the source
        """

        sources = []

        for location, source in self.sources.items():
            if not isinstance(source, source_type):
                continue

            if function is not None:
                function(source, location)
            else:
                sources.append((source, location))

        if function is None:
            return sources

    def update_source_for(
        self,
        location: ConvertableToLocation,
        source: BatchDataSource
        | BatchSourceable
        | ConvertableToRetrievalJob
        | RetrievalJob,
    ) -> ContractStore:
        location = convert_to_location(location)

        new_source = copy(self.sources)
        if isinstance(source, BatchDataSource):
            new_source[location] = source
        elif isinstance(source, pl.DataFrame):
            new_source[location] = InMemorySource(source)
        elif isinstance(source, RetrievalJob):
            new_source[location] = RetrievalJobSource(source)
        else:
            try:
                new_source[location] = resolve_source(source)  # type: ignore
            except ValueError:
                new_source[location] = InMemorySource.from_values(source)  # type: ignore

        return ContractStore(
            feature_views=self.feature_views, models=self.models, sources=new_source
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

    def write_request_for(self, location: FeatureLocation) -> RetrievalRequest:
        if location.location_type == "feature_view":
            return (
                self.feature_views[location.name]
                .request_all.needed_requests[0]
                .without_derived_features()
            )
        elif location.location_type == "model":
            return (
                self.models[location.name]
                .predictions_view.request("write", model_version_as_entity=True)
                .without_derived_features()
            )
        else:
            raise ValueError(f"Unable to write to location: '{location}'.")

    async def insert_into(
        self,
        location: ConvertableToLocation,
        values: ConvertableToRetrievalJob | RetrievalJob,
    ) -> None:
        used_location = convert_to_location(location)
        source = self.sources[used_location]

        write_request = self.write_request_for(used_location)

        if not isinstance(values, RetrievalJob):
            values = RetrievalJob.from_convertable(values, write_request)

        if isinstance(source, WritableFeatureSource):
            await source.insert(values, write_request)
        elif isinstance(source, DataFileReference):
            import polars as pl

            columns = write_request.all_returned_columns
            new_df = (await values.to_lazy_polars()).select(columns)

            try:
                existing_df = await source.to_lazy_polars()
                write_df = (
                    pl.concat(
                        [new_df, existing_df.select(columns)], how="vertical_relaxed"
                    )
                    .collect()
                    .lazy()
                )
            except (UnableToFindFileException, pl.exceptions.ComputeError):
                write_df = new_df

            if isinstance(source, ColumnFeatureMappable):
                new_cols = source.feature_identifier_for(columns)

                mappings = dict(zip(columns, new_cols))
                write_df = write_df.rename(mappings)

            await source.write_polars(write_df)
        else:
            raise ValueError(f"The source {type(source)} do not support writes")

    async def upsert_into(
        self,
        location: ConvertableToLocation,
        values: ConvertableToRetrievalJob | RetrievalJob,
    ) -> None:
        used_location = convert_to_location(location)
        source = self.sources[used_location]

        write_request = self.write_request_for(used_location)

        if not isinstance(values, RetrievalJob):
            values = RetrievalJob.from_convertable(values, write_request)

        if isinstance(source, WritableFeatureSource):
            await source.upsert(values, write_request)
        elif isinstance(source, DataFileReference):
            new_df = (await values.to_lazy_polars()).select(
                write_request.all_returned_columns
            )
            entities = list(write_request.entity_names)
            try:
                existing_df = await source.to_lazy_polars()
                write_df = upsert_on_column(entities, new_df, existing_df)
            except (UnableToFindFileException, pl.exceptions.ComputeError):
                write_df = new_df

            await source.write_polars(write_df)
        else:
            raise ValueError(f"The source {type(source)} do not support writes")

    async def overwrite(
        self,
        location: ConvertableToLocation,
        values: ConvertableToRetrievalJob | RetrievalJob,
        predicate: FilterRepresentable | None = None,
    ) -> None:
        used_location = convert_to_location(location)

        source = self.sources[used_location]
        write_request = self.write_request_for(used_location)

        exp_pred = None
        if predicate is not None:
            exp_pred = Expression.from_value(predicate)

        if not isinstance(values, RetrievalJob):
            values = RetrievalJob.from_convertable(values, write_request)

        if isinstance(source, WritableFeatureSource):
            await source.overwrite(values, write_request, predicate=exp_pred)
        elif isinstance(source, DataFileReference):
            df = (await values.to_lazy_polars()).select(
                write_request.all_returned_columns
            )
            if exp_pred is not None:
                raise NotImplementedError(
                    "Currently not implemented overwrite with predicate for DataFileReference"
                )

            await source.write_polars(df)
        else:
            raise ValueError(f"The source {type(source)} do not support writes")

    async def delete(
        self,
        location: ConvertableToLocation,
        predicate: FilterRepresentable | None = None,
    ) -> None:
        used_location = convert_to_location(location)

        source = self.sources[used_location]
        assert isinstance(source, Deletable)
        if predicate is not None:
            await source.delete(Expression.from_value(predicate))
        else:
            await source.delete()

    def needed_entities_for(self, features: list[FeatureReference]) -> set[Feature]:
        return self.requests_for_features(features).entities()


FeatureStore = ContractStore


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
    def location(self) -> FeatureLocation:
        return FeatureLocation.feature_view(self.name)

    @property
    def request(self) -> RetrievalRequest:
        if self.feature_filter is not None:
            features_in_models = self.store.model_features_for(self.view.name)
            logger.info(f"Only processing model features: {features_in_models}")
            return self.view.request_for(features_in_models).needed_requests[0]
        else:
            return self.view.request_all.needed_requests[0]

    @property
    def source(self) -> BatchDataSource:
        return self.store.sources[FeatureLocation.feature_view(self.name)]

    def using_source(
        self, source: BatchDataSource | BatchSourceable
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
        return FeatureViewStore(
            self.store.update_source_for(self.location, source),
            view=self.view,
            event_triggers=self.event_triggers,
            feature_filter=self.feature_filter,
        )

    def with_optimised_write(self) -> FeatureViewStore:
        features_in_models = self.store.model_features_for(self.view.name)
        return self.select(features_in_models)

    def select_columns(
        self, columns: Iterable[str | FeatureFactory], limit: int | None = None
    ) -> RetrievalJob:
        return self.select(set(columns)).all(limit)

    def all(self, limit: int | None = None) -> RetrievalJob:
        return self.all_columns(limit)

    def filter(self, filter: pl.Expr | str | FeatureFactory) -> RetrievalJob:
        return self.all().filter(filter)

    @overload
    def drop_invalid(
        self, values: pl.LazyFrame, validator: Validator | None = None
    ) -> pl.LazyFrame: ...

    @overload
    def drop_invalid(
        self, values: pl.DataFrame, validator: Validator | None = None
    ) -> pl.DataFrame: ...

    @overload
    def drop_invalid(
        self, values: pd.DataFrame, validator: Validator | None = None
    ) -> pd.DataFrame: ...

    @overload
    def drop_invalid(
        self, values: dict[str, list], validator: Validator | None = None
    ) -> dict[str, list]: ...

    @overload
    def drop_invalid(
        self, values: list[dict[str, Any]], validator: Validator | None = None
    ) -> list[dict[str, Any]]: ...

    def drop_invalid(
        self, values: ConvertableToRetrievalJob, validator: Validator | None = None
    ) -> ConvertableToRetrievalJob:
        from aligned.retrieval_job import DropInvalidJob

        if validator is None:
            validator = PolarsValidator()

        features = list(DropInvalidJob.features_to_validate([self.request]))

        if isinstance(values, pl.LazyFrame):
            return validator.validate_polars(features, values)
        elif isinstance(values, pl.DataFrame):
            df = values.lazy()
            return validator.validate_polars(features, df).collect()
        elif isinstance(values, dict):
            df = pl.DataFrame(values).lazy()
            return (
                validator.validate_polars(features, df)
                .collect()
                .to_dict(as_series=False)
            )
        elif isinstance(values, list):
            df = pl.DataFrame(values).lazy()
            return validator.validate_polars(features, df).collect().to_dicts()
        elif isinstance(values, pd.DataFrame):
            df = pl.from_pandas(values).lazy()
            return validator.validate_polars(features, df).collect().to_pandas()
        else:
            raise ValueError(f"Unable to convert {type(values)}")

    def all_columns(self, limit: int | None = None) -> RetrievalJob:
        request = self.view.request_all
        if self.feature_filter:
            request = self.view.request_for(self.feature_filter)

        assert len(request.needed_requests) == 1

        job = (
            self.source.all_data(request.needed_requests[0], limit)
            .ensure_types(request.needed_requests)
            .derive_features(request.needed_requests)
            .inject_store(self.store)
        )
        if self.feature_filter:
            selected_columns = self.feature_filter
        else:
            selected_columns = set(request.needed_requests[0].all_returned_columns)

        return job.select_columns(selected_columns)

    def between_dates(self, start_date: datetime, end_date: datetime) -> RetrievalJob:
        if self.feature_filter:
            request = self.view.request_for(self.feature_filter)
            return SelectColumnsJob(
                list(self.feature_filter),
                self.source.all_between_dates(
                    request.needed_requests[0], start_date, end_date
                )
                .ensure_types()
                .derive_features()
                .inject_store(self.store),
            )

        request = self.view.request_all
        return (
            self.source.all_between_dates(
                request.needed_requests[0], start_date, end_date
            )
            .ensure_types()
            .derive_features()
            .inject_store(self.store)
        )

    def previous(
        self, days: int = 0, minutes: int = 0, seconds: int = 0
    ) -> RetrievalJob:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days, minutes=minutes, seconds=seconds)
        return self.between_dates(start_date, end_date)

    def features_for(
        self,
        entities: ConvertableToRetrievalJob | RetrievalJob,
        event_timestamp_column: str | None = None,
    ) -> RetrievalJob:
        features = {"*"}
        if self.feature_filter:
            features = self.feature_filter

        feature_refs = [f"{self.view.name}:{feature}" for feature in features]
        return self.store.features_for(
            entities,
            feature_refs,
            event_timestamp_column=event_timestamp_column,
        )

    def select(self, features: Iterable[str | FeatureFactory]) -> FeatureViewStore:
        logger.info(f"Selecting features {features}")
        return FeatureViewStore(
            self.store,
            self.view,
            self.event_triggers,
            {feat if isinstance(feat, str) else feat.name for feat in features},
        )

    async def upsert(self, values: RetrievalJob | ConvertableToRetrievalJob) -> None:
        await self.store.upsert_into(FeatureLocation.feature_view(self.name), values)

    async def insert(self, values: RetrievalJob | ConvertableToRetrievalJob) -> None:
        await self.store.insert_into(FeatureLocation.feature_view(self.name), values)

    async def overwrite(
        self,
        values: RetrievalJob | ConvertableToRetrievalJob,
        predicate: FilterRepresentable | None = None,
    ) -> None:
        await self.store.overwrite(
            FeatureLocation.feature_view(self.name), values, predicate
        )

    async def delete(self, predicate: FilterRepresentable | None = None) -> None:
        await self.store.delete(FeatureLocation.feature_view(self.name), predicate)

    @property
    def write_input(self) -> set[str]:
        features = set()
        for request in self.view.request_all.needed_requests:
            features.update(request.all_required_feature_names)
            features.update(request.entity_names)
            if event_timestamp := request.event_timestamp:
                features.add(event_timestamp.name)
        return features

    async def write(self, values: ConvertableToRetrievalJob) -> None:
        from aligned import FileSource
        from aligned.schemas.derivied_feature import AggregateOver

        requests = self.view.request_all.needed_requests
        if self.feature_filter is not None:
            logger.info(f"Filtering features to {self.feature_filter}")
            requests = self.view.request_for(self.feature_filter).needed_requests

        if len(requests) != 1:
            raise ValueError(
                f"Something odd happened. Expected 1 request when writing, got {len(requests)}"
            )

        request = requests[0]

        job = (
            RetrievalJob.from_convertable(values, request)
            .validate_entites()
            .fill_missing_columns()
            .ensure_types([request])
        )

        aggregations = request.aggregate_over()
        if aggregations:
            checkpoints: dict[AggregateOver, DataFileReference] = {}

            for aggregation in aggregations.keys():
                name = f"{self.view.name}_agg"

                if aggregation.window:
                    name += f"_{aggregation.window.time_window.total_seconds()}"

                if aggregation.condition:
                    name += f"_{aggregation.condition.name}"

                checkpoints[aggregation] = FileSource.parquet_at(name)

            job = StreamAggregationJob(job, checkpoints)

        job = job.derive_features([request])

        if self.feature_filter:
            job = job.select_columns(self.feature_filter)

        await self.batch_write(job)

    def process_input(self, values: ConvertableToRetrievalJob) -> RetrievalJob:
        request = self.view.request_all.needed_requests[0]

        job = RetrievalJob.from_convertable(values, request)

        return (
            job.fill_missing_columns()
            .ensure_types([request])
            .aggregate(request)
            .derive_features([request])
        )

    async def batch_write(
        self, values: ConvertableToRetrievalJob | RetrievalJob
    ) -> None:
        """Takes a set of features, computes the derived features, and store them in the source

        Args:
            values (dict[str, list[Any]] | RetrievalJob): The features to write

        Raises:
            ValueError: In case the inputted features are invalid
        """

        if not isinstance(self.source, WritableFeatureSource):
            logger.info("Feature Source is not writable")
            return

        # As it is a feature view, should it only contain one request
        request = self.request

        core_job: RetrievalJob

        if isinstance(values, RetrievalJob):
            core_job = values
        else:
            core_job = RetrievalJob.from_convertable(values, request)

        # job = (
        #     core_job.derive_features([request])
        #     .listen_to_events(self.event_triggers)
        #     .update_vector_index(self.view.indexes)
        # )
        job = core_job

        # if self.feature_filter:
        #     logger.info(f'Only writing features used by models: {self.feature_filter}')
        #     job = job.filter(self.feature_filter)

        await self.source.insert(job, job.retrieval_requests[0])

    async def freshness(self) -> datetime | None:
        view = self.view
        if not view.event_timestamp:
            raise ValueError(
                f"View named '{view.name}' have no event timestamp. Therefore, unable to compute freshness"
            )

        return await self.source.freshness(view.event_timestamp.as_feature())


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
        self, entities: RetrievalJob | ConvertableToRetrievalJob, number_of_records: int
    ) -> RetrievalJob:
        source = self.model.predictions_view.source
        assert isinstance(source, VectorIndex)

        embeddings = self.model.predictions_view.embeddings()
        n_embeddings = len(embeddings)

        if n_embeddings == 0:
            raise ValueError(
                f"Need at least one embedding to search. Got {n_embeddings}"
            )
        if n_embeddings > 1:
            raise ValueError(
                "Got more than one embedding, it is therefore unclear which to use."
            )

        embedding = embeddings[0]
        response = self.model.predictions_view.request(self.model.name)

        def contains_embedding() -> bool:
            if isinstance(entities, RetrievalJob):
                return embedding.name in entities.loaded_columns
            elif isinstance(entities, dict):
                return embedding.name in entities
            elif isinstance(entities, (pl.DataFrame, pd.DataFrame, pl.LazyFrame)):
                return embedding.name in entities.columns
            raise ValueError(
                "Unable to determine if the entities contains the embedding"
            )

        if self.model.exposed_model and not contains_embedding():
            model_store = self.store.model(self.model.name)
            features: RetrievalJob = model_store.predict_over(entities)
        else:
            # Assumes that we can lookup the embeddings from the source
            feature_ref = FeatureReference(
                embedding.name, FeatureLocation.model(self.model.name)
            )
            features: RetrievalJob = self.store.features_for(
                entities, features=[feature_ref.identifier]
            )

        return source.nearest_n_to(features, number_of_records, response)

    def as_langchain_retriver(self, number_of_docs: int = 5):
        from aligned.exposed_model.langchain_retriever import AlignedRetriever

        return AlignedRetriever(
            store=self.store, index_name=self.index_name, number_of_docs=number_of_docs
        )
