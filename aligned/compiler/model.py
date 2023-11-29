from __future__ import annotations

import copy
import logging
from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Type, TypeVar, Generic

from uuid import uuid4

import polars as pl

from aligned.compiler.feature_factory import (
    ClassificationLabel,
    Entity,
    Bool,
    EventTimestamp,
    FeatureFactory,
    FeatureReferencable,
    RegressionLabel,
    TargetProbability,
    ModelVersion,
)
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.entity_data_source import EntityDataSource
from aligned.feature_view.feature_view import FeatureView
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReferance, FeatureType
from aligned.schemas.folder import Folder
from aligned.schemas.literal_value import LiteralValue
from aligned.schemas.model import Model as ModelSchema
from aligned.schemas.model import PredictionsView
from aligned.schemas.target import ClassificationTarget as ClassificationTargetSchema
from aligned.schemas.target import RegressionTarget as RegressionTargetSchema

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SqlEntityDataSource(EntityDataSource):

    url: str
    timestamp_column: str

    def __init__(self, sql: Callable[[str], str], url: str, timestamp_column: str) -> None:
        self.sql = sql
        self.url = url
        self.timestamp_column = timestamp_column

    async def all_in_range(self, start_date: datetime, end_date: datetime) -> pl.DataFrame:
        import os

        start = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end = end_date.strftime('%Y-%m-%d %H:%M:%S')

        query = self.sql(f'{self.timestamp_column} BETWEEN \'{start}\' AND \'{end}\'')
        return pl.read_sql(query, os.environ[self.url])

    async def last(self, days: int, hours: int, seconds: int) -> pl.DataFrame:
        now = datetime.utcnow()
        return await self.all_in_range(now - timedelta(days=days, hours=hours, seconds=seconds), now)


@dataclass
class ModelMetadata:
    name: str
    features: list[FeatureReferencable]
    # Will log the feature inputs to a model. Therefore, enabling log and wait etc.
    # feature_logger: WritableBatchSource | None = field(default=None)
    contacts: list[str] | None = field(default=None)
    tags: dict[str, str] | None = field(default=None)
    description: str | None = field(default=None)
    prediction_source: BatchDataSource | None = field(default=None)
    prediction_stream: StreamDataSource | None = field(default=None)
    application_source: BatchDataSource | None = field(default=None)
    dataset_folder: Folder | None = field(default=None)


@dataclass
class ModelContractWrapper(Generic[T]):

    metadata: ModelMetadata
    contract: Type[T]

    def __call__(self) -> T:
        # Needs to compiile the model to set the location for the view features
        _ = self.compile()

        # Need to copy and set location in case filters are used.
        # As this can lead to incorrect features otherwise
        contract = copy.deepcopy(self.contract())
        for attribute in dir(contract):
            if attribute.startswith('__'):
                continue

            value = getattr(contract, attribute)
            if isinstance(value, FeatureFactory):
                value._location = FeatureLocation.model(self.metadata.name)
                setattr(contract, attribute, copy.deepcopy(value))

        return contract

    def compile(self) -> ModelSchema:
        return ModelContract.compile_with_metadata(self.contract(), self.metadata)

    def filter(
        self, name: str, where: Callable[[T], Bool], application_source: BatchDataSource | None = None
    ) -> ModelContractWrapper[T]:
        from aligned.data_source.batch_data_source import FilteredDataSource

        meta = self.metadata
        meta.name = name

        condition = where(self.__call__())

        main_source = meta.prediction_source
        if not main_source:
            raise ValueError(
                f'Model: {self.metadata.name} needs a `prediction_source` to use `filter`, got None.'
            )

        if not condition._name:
            condition._name = str(uuid4())
            condition._location = FeatureLocation.model(name)

        if condition.transformation:
            meta.prediction_source = FilteredDataSource(main_source, condition.compile())
        else:
            meta.prediction_source = FilteredDataSource(main_source, condition.feature())

        if application_source:
            meta.application_source = application_source

        return ModelContractWrapper(metadata=meta, contract=self.contract)


def model_contract(
    name: str,
    features: list[FeatureReferencable],
    contacts: list[str] | None = None,
    tags: dict[str, str] | None = None,
    description: str | None = None,
    prediction_source: BatchDataSource | None = None,
    prediction_stream: StreamDataSource | None = None,
    application_source: BatchDataSource | None = None,
    dataset_folder: Folder | None = None,
) -> Callable[[Type[T]], ModelContractWrapper[T]]:
    def decorator(cls: Type[T]) -> ModelContractWrapper[T]:
        metadata = ModelMetadata(
            name,
            features,
            contacts=contacts,
            tags=tags,
            description=description,
            prediction_source=prediction_source,
            prediction_stream=prediction_stream,
            application_source=application_source,
            dataset_folder=dataset_folder,
        )
        return ModelContractWrapper(metadata, cls)

    return decorator


class ModelContract(ABC):
    @staticmethod
    def metadata_with(
        name: str,
        features: list[FeatureReferencable],
        description: str | None = None,
        contacts: list[str] | None = None,
        tags: dict[str, str] | None = None,
        predictions_source: BatchDataSource | None = None,
        predictions_stream: StreamDataSource | None = None,
        application_source: BatchDataSource | None = None,
        dataset_folder: Folder | None = None,
    ) -> ModelMetadata:
        return ModelMetadata(
            name,
            features,
            contacts,
            tags,
            description,
            predictions_source,
            predictions_stream,
            application_source=application_source,
            dataset_folder=dataset_folder,
        )

    @abstractproperty
    def metadata(self) -> ModelMetadata:
        raise NotImplementedError()

    @classmethod
    def compile(cls) -> ModelSchema:
        instance = cls()
        return ModelContract.compile_with_metadata(instance, instance.metadata)

    @staticmethod
    def compile_with_metadata(model: Any, metadata: ModelMetadata) -> ModelSchema:
        """
        Compiles the ModelContract in to ModelSchema structure that can further be encoded.

        ```python
        class MyModel(ModelContract):
            ...

            metadata = ModelContract.metadata_with(...)

        model_schema = MyModel().compile_instance()

        ```

        Returns: The compiled Model schema
        """
        var_names = [name for name in model.__dir__() if not name.startswith('_')]

        inference_view: PredictionsView = PredictionsView(
            entities=set(),
            features=set(),
            derived_features=set(),
            model_version_column=None,
            source=metadata.prediction_source,
            application_source=metadata.application_source,
            stream_source=metadata.prediction_stream,
            classification_targets=set(),
            regression_targets=set(),
        )
        probability_features: dict[str, set[TargetProbability]] = {}

        classification_targets: dict[str, ClassificationTargetSchema] = {}
        regression_targets: dict[str, RegressionTargetSchema] = {}

        for var_name in var_names:
            feature = getattr(model, var_name)

            if isinstance(feature, FeatureFactory):
                feature._location = FeatureLocation.model(metadata.name)

            if isinstance(feature, ModelVersion):
                inference_view.model_version_column = feature.feature()
            if isinstance(feature, FeatureView):
                compiled = feature.compile()
                inference_view.entities.update(compiled.entities)
            elif isinstance(feature, ModelContract):
                compiled = feature.compile()
                inference_view.entities.update(compiled.predictions_view.entities)
            elif isinstance(feature, ClassificationLabel):
                assert feature._name
                feature._location = FeatureLocation.model(metadata.name)
                target_feature = feature.compile()

                classification_targets[var_name] = target_feature
                inference_view.classification_targets.add(target_feature)
            elif isinstance(feature, RegressionLabel):
                assert feature._name
                feature._location = FeatureLocation.model(metadata.name)
                target_feature = feature.compile()

                regression_targets[var_name] = target_feature
                inference_view.regression_targets.add(target_feature)
                inference_view.features.add(target_feature.feature)
            elif isinstance(feature, EventTimestamp):
                inference_view.event_timestamp = feature.event_timestamp()

            elif isinstance(feature, TargetProbability):
                feature_name = feature.target._name
                assert feature._name
                assert (
                    feature.target._name in classification_targets
                ), 'Target must be a classification target.'

                target = classification_targets[feature.target._name]
                target.class_probabilities.add(feature.compile())

                inference_view.features.add(
                    Feature(
                        var_name,
                        FeatureType.float(),
                        f"The probability of target named {feature_name} being '{feature.of_value}'.",
                    )
                )
                probability_features[feature_name] = probability_features.get(feature_name, set()).union(
                    {feature}
                )
            elif isinstance(feature, Entity):
                inference_view.entities.add(feature.feature())
            elif isinstance(feature, FeatureFactory):
                inference_view.features.add(feature.feature())

        # Needs to run after the feature views have compiled
        features: set[FeatureReferance] = {feature.feature_referance() for feature in metadata.features}

        for target, probabilities in probability_features.items():
            from aligned.schemas.transformation import MapArgMax

            transformation = MapArgMax(
                {probs._name: LiteralValue.from_value(probs.of_value) for probs in probabilities}
            )

            arg_max_feature = DerivedFeature(
                name=target,
                dtype=transformation.dtype,
                transformation=transformation,
                depending_on={
                    FeatureReferance(feat, FeatureLocation.model(metadata.name), dtype=FeatureType.float())
                    for feat in transformation.column_mappings.keys()
                },
                depth=1,
            )
            inference_view.derived_features.add(arg_max_feature)

        if not probability_features and inference_view.classification_targets:
            inference_view.features.update(
                {target.feature for target in inference_view.classification_targets}
            )

        return ModelSchema(
            name=metadata.name,
            features=features,
            predictions_view=inference_view,
            contacts=metadata.contacts,
            tags=metadata.tags,
            description=metadata.description,
            dataset_folder=metadata.dataset_folder,
        )
