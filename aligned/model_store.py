from __future__ import annotations
from typing import TYPE_CHECKING

from aligned.exposed_model.interface import ExposedModel

import polars as pl

from aligned.lazy_imports import pandas as pd

import logging
from dataclasses import dataclass
from datetime import datetime

from aligned.data_source.batch_data_source import (
    BatchDataSource,
)
from aligned.request.retrieval_request import FeatureRequest, RetrievalRequest
from aligned.retrieval_job import (
    FilterRepresentable,
    PredictionJob,
    RetrievalJob,
    SupervisedJob,
    ConvertableToRetrievalJob,
)
from aligned.schemas.feature import FeatureLocation, Feature, FeatureReference
from aligned.schemas.folder import DatasetStore
from aligned.schemas.model import Model as ModelSchema

if TYPE_CHECKING:
    from aligned.feature_store import ContractStore, DataFileReference

logger = logging.getLogger(__name__)


@dataclass
class ModelFeatureStore:
    model: ModelSchema
    store: ContractStore
    selected_version: str | None = None
    model_to_use: ExposedModel | None = None

    @property
    def location(self) -> FeatureLocation:
        return FeatureLocation.model(self.model.name)

    @property
    def dataset_store(self) -> DatasetStore | None:
        return self.model.dataset_store

    @property
    def source(self) -> BatchDataSource:
        return self.store.sources[self.location]

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

    def input_features(self) -> list[FeatureReference]:
        """
        The input features to load, based on the input features in a model.
        """
        version = self.selected_version or self.model.features.default_version
        return self.model.features.features_for(version)

    def raw_string_features(self, except_features: set[str]) -> set[str]:
        features = self.input_features()
        return {
            f"{feature.location.identifier}:{feature.name}"
            for feature in features
            if feature.name not in except_features
        }

    def using_version(self, version: str) -> ModelFeatureStore:
        return ModelFeatureStore(self.model, self.store, version)

    def request(
        self,
        except_features: set[str] | None = None,
        event_timestamp_column: str | None = None,
    ) -> FeatureRequest:
        return self.input_request(except_features, event_timestamp_column)

    def input_request(
        self,
        except_features: set[str] | None = None,
        event_timestamp_column: str | None = None,
    ) -> FeatureRequest:
        feature_refs = self.raw_string_features(except_features or set())
        if not feature_refs:
            raise ValueError(f"No features to request for model '{self.model.name}'")

        return self.store.requests_for_features(
            list(feature_refs),
            event_timestamp_column,
        )

    def prediction_request(
        self,
        exclude_features: set[str] | None = None,
        model_version_as_entity: bool = False,
    ) -> RetrievalRequest:
        if not self.model.predictions_view:
            raise ValueError(f"Model {self.model.name} has no predictions view")

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

    def using_model(self, model_to_use: ExposedModel) -> ModelFeatureStore:
        """
        Makes it possible to define which model to use for predictions.
        """
        return ModelFeatureStore(
            self.model, self.store, self.selected_version, model_to_use
        )

    def predict_over(
        self,
        entities: ConvertableToRetrievalJob | RetrievalJob,
    ) -> PredictionJob:
        predictor = self.model_to_use or self.model.exposed_model
        if not predictor:
            raise ValueError(
                f"Model {self.model.name} has no predictor set. "
                "This can be done by setting the `exposed_at` value"
            )

        returned_request = self.request().needed_requests

        if not isinstance(entities, RetrievalJob):
            entities = RetrievalJob.from_convertable(entities, returned_request)

        return PredictionJob(
            entities,
            self.model,
            self.store,
            predictor=predictor,
            output_requests=returned_request,
        )

    def input_features_for(
        self,
        entities: ConvertableToRetrievalJob | RetrievalJob,
        event_timestamp_column: str | None = None,
    ) -> RetrievalJob:
        """Returns the features to pass into a model

        ```python
        store = await FileSource.json_at("contracts.json").feature_store()

        df = store.model("titanic")\\
            .input_features_for({"passenger_id": [1, 2, 3]})\\
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
            entities (dict[str, list] | RetrievalJob): The entities to fetch features for

        Returns:
            RetrievalJob: A retrieval job that can be used to fetch the features
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
                    RetrievalJob.from_convertable(entities, request)
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

            if loc.location_type == "model":
                event_timestamp = (
                    self.store.model(loc.name)
                    .model.predictions_view.as_view(loc.name)
                    .freshness_feature
                )
            else:
                event_timestamp = self.store.feature_view(
                    loc.name
                ).view.freshness_feature

            if event_timestamp:
                locs[loc] = event_timestamp

        freshnesses: dict[FeatureLocation, datetime | None] = {}
        for loc in locs.keys():
            freshnesses[loc] = await self.store.freshness_for(loc)

        return freshnesses

    async def prediction_freshness(self) -> datetime | None:
        return await self.store.freshness_for(self.location)

    def with_labels(
        self, label_refs: set[FeatureReference] | None = None
    ) -> SupervisedModelFeatureStore:
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

    def cached_at(self, location: DataFileReference) -> RetrievalJob:
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
            RetrievalJob: _description_
        """
        from aligned.local.job import FileFullJob

        references = self.model.feature_references(self.selected_version)
        features = {
            f"{feature.location.identifier}:{feature.name}" for feature in references
        }
        request = self.store.requests_for_features(list(features))

        return FileFullJob(
            location, RetrievalRequest.unsafe_combine(request.needed_requests)
        ).select_columns(request.features_to_include)

    def process_features(
        self, input: RetrievalJob | ConvertableToRetrievalJob
    ) -> RetrievalJob:
        request = self.request()

        if isinstance(input, RetrievalJob):
            job = input.select_columns(request.features_to_include)
        else:
            job = RetrievalJob.from_convertable(input, request=request.needed_requests)

        return (
            job.ensure_types(request.needed_requests)
            .derive_features(request.needed_requests)
            .select_columns(request.features_to_include)
        )

    def features_for(
        self,
        entities: ConvertableToRetrievalJob | RetrievalJob,
        event_timestamp_column: str | None = None,
        model_version_as_entity: bool | None = None,
    ) -> RetrievalJob:
        """Returns the features the model has produced.

        ```python
        store = await FileSource.json_at("contracts.json").feature_store()

        df = store.model("titanic")\\
            .features_for({"passenger_id": [1, 2, 3]})\\
            .to_polars()

        print(df.collect())
        >>> ┌──────────────┬──────────────┐
        >>> │ passenger_id ┆ will_survive │
        >>> │ ---          ┆ ---          │
        >>> │ i32          ┆ bool         │
        >>> ╞══════════════╪══════════════╡
        >>> │ 1            ┆ true         │
        >>> │ 2            ┆ true         │
        >>> │ 3            ┆ false        │
        >>> └──────────────┴──────────────┘
        ```

        Args:
            entities (dict[str, list] | RetrievalJob): The entities to fetch features for

        Returns:
            RetrievalJob: A retrieval job that can be used to fetch the features
        """
        location_id = self.location.identifier
        return self.store.features_for(
            entities,
            features=[f"{location_id}:*"],
            event_timestamp_column=event_timestamp_column,
            model_version_as_entity=model_version_as_entity,
        )

    def predictions_for(
        self,
        entities: ConvertableToRetrievalJob | RetrievalJob,
        event_timestamp_column: str | None = None,
        model_version_as_entity: bool | None = None,
    ) -> RetrievalJob:
        return self.features_for(
            entities,
            event_timestamp_column=event_timestamp_column,
            model_version_as_entity=model_version_as_entity,
        )

    def predictions_between(
        self, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        source = self.source
        request = self.model.predictions_view.request(self.model.name)

        return source.all_between_dates(request, start_date, end_date).select_columns(
            set(request.all_returned_columns)
        )

    def all_predictions(self, limit: int | None = None) -> RetrievalJob:
        source = self.source
        request = self.model.predictions_view.request(self.model.name)

        return (
            source.all_data(request, limit=limit)
            .inject_store(self.store)
            .select_columns(set(request.all_returned_columns))
        )

    def filter_predictions(self, filter: FilterRepresentable) -> RetrievalJob:
        return self.all_predictions().filter(filter)

    def using_source(self, source: BatchDataSource) -> ModelFeatureStore:
        return ModelFeatureStore(
            self.model, self.store.update_source_for(self.location, source)
        )

    def depends_on(self) -> set[FeatureLocation]:
        """
        Returns the views and models that the model depend on to compute it's output.

        Examples:
            ```python
            @data_contract(name="passenger", ...)
            class Passenger:
                passenger_id = Int32().as_entity()

                age = Float()

            @data_contract(name="location", ...)
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
        try:
            locs = {req.location for req in self.request().needed_requests}
        except ValueError:
            locs = set()

            if self.model.predictions_view.source:
                for dep in self.model.predictions_view.source.depends_on():
                    locs.add(dep)

        label_refs = self.model.predictions_view.labels_estimates_refs()
        if label_refs:
            for ref in label_refs:
                locs.add(ref.location)

        for conf in self.model.predictions_view.recommendation_targets or set():
            if conf.was_selected_view:
                locs.add(conf.was_selected_view)

        return locs

    async def overwrite(self, output: ConvertableToRetrievalJob | RetrievalJob) -> None:
        """
        Overwrites data to a source defined as a output source

        ```python
        @model_contract(
            name="taxi_eta",
            features=[...]
            output_source=FileSource.parquet_at("predictions.parquet")
        )
        class TaxiEta:
            trip_id = Int32().as_entity()

            duration = Int32()

        ...

        store = FeatureStore.from_dir(".")

        await store.model("taxi_eta").overwrite({
            "trip_id": [1, 2, 3, ...],
            "duration": [20, 33, 42, ...]
        })
        ```
        """
        await self.store.overwrite(FeatureLocation.model(self.model.name), output)

    async def upsert(self, output: ConvertableToRetrievalJob | RetrievalJob) -> None:
        """
        Upserts data to a source defined as a output source

        ```python
        @model_contract(
            name="taxi_eta",
            features=[...]
            output_source=FileSource.parquet_at("predictions.parquet")
        )
        class TaxiEta:
            trip_id = Int32().as_entity()

            duration = Int32()

        ...

        store = FeatureStore.from_dir(".")

        await store.model("taxi_eta").upsert({
            "trip_id": [1, 2, 3, ...],
            "duration": [20, 33, 42, ...]
        })
        ```
        """
        await self.store.upsert_into(FeatureLocation.model(self.model.name), output)

    async def upsert_predictions(
        self, predictions: ConvertableToRetrievalJob | RetrievalJob
    ) -> None:
        """
        Upserts data to a source defined as a output source

        ```python
        @model_contract(
            name="taxi_eta",
            features=[...]
            output_source=FileSource.parquet_at("predictions.parquet")
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
        await self.upsert(predictions)

    async def insert(self, output: ConvertableToRetrievalJob | RetrievalJob) -> None:
        """
        Writes data to a source defined as a output source

        ```python
        @model_contract(
            name="taxi_eta",
            features=[...]
            output_source=FileSource.parquet_at("predictions.parquet")
        )
        class TaxiEta:
            trip_id = Int32().as_entity()

            duration = Int32()

        ...

        store = FeatureStore.from_dir(".")

        await store.model("taxi_eta").insert({
            "trip_id": [1, 2, 3, ...],
            "duration": [20, 33, 42, ...]
        })
        ```
        """
        await self.store.insert_into(FeatureLocation.model(self.model.name), output)

    async def insert_predictions(
        self, predictions: ConvertableToRetrievalJob | RetrievalJob
    ) -> None:
        """
        Writes data to a source defined as a output source

        ```python
        @model_contract(
            name="taxi_eta",
            features=[...]
            output_source=FileSource.parquet_at("predictions.parquet")
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
        await self.insert(predictions)


@dataclass
class SupervisedModelFeatureStore:
    model: ModelSchema
    store: ContractStore
    labels_estimates_refs: set[FeatureReference]

    selected_version: str | None = None

    def features_for(
        self,
        entities: ConvertableToRetrievalJob | RetrievalJob,
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
            entities (dict[str, list] | RetrievalJob): A dictionary of entity names to lists of entity values

        Returns:
            SupervisedJob: A object that will load the features and labels in your desired format
        """
        feature_refs = self.model.feature_references(self.selected_version)
        features = {
            f"{feature.location.identifier}:{feature.name}" for feature in feature_refs
        }
        pred_view = self.model.predictions_view

        target_feature_refs = self.labels_estimates_refs
        target_features = {feature.identifier for feature in target_feature_refs}

        targets = set()
        if pred_view.classification_targets:
            targets = {
                feature.estimating.name for feature in pred_view.classification_targets
            }
        elif pred_view.regression_targets:
            targets = {
                feature.estimating.name for feature in pred_view.regression_targets
            }
        else:
            raise ValueError("Found no targets in the model")

        if event_timestamp_column == target_event_timestamp_column:
            request = self.store.requests_for_features(
                list(features.union(target_features)),
                event_timestamp_column=event_timestamp_column,
            )
            job = self.store.features_for_request(
                request, entities, request.features_to_include
            )
            return SupervisedJob(
                job.select_columns(request.features_to_include),
                target_columns=targets,
            )

        request = self.store.requests_for_features(
            list(features), event_timestamp_column=event_timestamp_column
        )
        target_request = self.store.requests_for_features(
            list(target_features), event_timestamp_column=target_event_timestamp_column
        ).with_sufix("target")

        total_request = FeatureRequest(
            FeatureLocation.model(self.model.name),
            request.features_to_include.union(target_request.features_to_include),
            request.needed_requests + target_request.needed_requests,
        )
        job = self.store.features_for_request(
            total_request, entities, total_request.features_to_include
        )
        return SupervisedJob(
            job.select_columns(total_request.features_to_include),
            target_columns=targets,
        )

    def predictions_for(
        self,
        entities: ConvertableToRetrievalJob | RetrievalJob,
        event_timestamp_column: str | None = None,
        target_event_timestamp_column: str | None = None,
    ) -> RetrievalJob:
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
            entities (dict[str, list] | RetrievalJob): A dictionary of entity names to lists of entity values

        Returns:
            RetrievalJob: A object that will load the features and labels in your desired format
        """

        pred_view = self.model.predictions_view
        if pred_view.source is None:
            raise ValueError(
                "Model does not have a prediction source. "
                "This can be set in the metadata for a model contract."
            )

        request = pred_view.request(self.model.name)

        target_features = pred_view.labels_estimates_refs()
        target_features = {feature.identifier for feature in target_features}

        labels = pred_view.labels()
        pred_features = {
            f"model:{self.model.name}:{feature.name}" for feature in labels
        }
        request = self.store.requests_for_features(
            list(pred_features), event_timestamp_column=event_timestamp_column
        )
        target_request = self.store.requests_for_features(
            list(target_features),
            event_timestamp_column=target_event_timestamp_column,
        ).with_sufix("target")

        total_request = FeatureRequest(
            FeatureLocation.model(self.model.name),
            request.features_to_include.union(target_request.features_to_include),
            request.needed_requests + target_request.needed_requests,
        )
        return self.store.features_for_request(
            total_request, entities, total_request.features_to_include
        )
