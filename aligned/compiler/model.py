from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Type, TypeVar, Generic, TYPE_CHECKING
from datetime import timedelta

from uuid import uuid4

from aligned.compiler.feature_factory import (
    ClassificationLabel,
    Entity,
    Bool,
    EventTimestamp,
    FeatureFactory,
    FeatureReferencable,
    RecommendationTarget,
    RegressionLabel,
    TargetProbability,
    ModelVersion,
)
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.feature_view.feature_view import FeatureView, FeatureViewWrapper
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReferance, FeatureType
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.schemas.literal_value import LiteralValue
from aligned.schemas.model import Model as ModelSchema
from aligned.schemas.model import FeatureInputVersions as FeatureVersionSchema
from aligned.schemas.model import PredictionsView
from aligned.schemas.target import ClassificationTarget as ClassificationTargetSchema
from aligned.schemas.target import RegressionTarget as RegressionTargetSchema

if TYPE_CHECKING:
    from aligned.sources.local import StorageFileReference
    from aligned.schemas.folder import DatasetStore

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ModelMetadata:
    name: str
    features: FeatureInputVersions
    # Will log the feature inputs to a model. Therefore, enabling log and wait etc.
    # feature_logger: WritableBatchSource | None = field(default=None)
    contacts: list[str] | None = field(default=None)
    tags: list[str] | None = field(default=None)
    description: str | None = field(default=None)
    prediction_source: BatchDataSource | None = field(default=None)
    prediction_stream: StreamDataSource | None = field(default=None)
    application_source: BatchDataSource | None = field(default=None)

    acceptable_freshness: timedelta | None = field(default=None)
    unacceptable_freshness: timedelta | None = field(default=None)

    exposed_at_url: str | None = field(default=None)

    dataset_store: DatasetStore | None = field(default=None)


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

        setattr(contract, '__model_wrapper__', self)
        return contract

    def compile(self) -> ModelSchema:
        return compile_with_metadata(self.contract(), self.metadata)

    def as_view(self) -> CompiledFeatureView | None:

        compiled = self.compile()
        view = compiled.predictions_view

        if not view.source:
            return None

        return CompiledFeatureView(
            name=self.metadata.name,
            source=view.source,
            entities=view.entities,
            features=view.features,
            derived_features=view.derived_features,
            event_timestamp=view.event_timestamp,
        )

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

    def as_source(self) -> BatchDataSource:
        from aligned.schemas.model import ModelSource

        compiled_model = self.compile()
        compiled_view = self.as_view()

        if compiled_view is None:
            raise ValueError(f"Model {compiled_model.name} is not compiled as a view")

        return ModelSource(compiled_model, compiled_view)

    def join(
        self,
        view: FeatureViewWrapper,
        on_left: str | FeatureFactory | list[str] | list[FeatureFactory],
        on_right: str | FeatureFactory | list[str] | list[FeatureFactory],
        how: str = 'inner',
    ) -> BatchDataSource:
        from aligned.data_source.batch_data_source import join_source
        from aligned.schemas.model import ModelSource

        compiled_model = self.compile()
        compiled_view = self.as_view()

        if compiled_view is None:
            raise ValueError(f"Model {compiled_model.name} is not compiled as a view")

        source = ModelSource(compiled_model, compiled_view)

        return join_source(
            source,
            view=view,
            on_left=on_left,
            on_right=on_right,
            left_request=compiled_view.request_all.needed_requests[0],
            how=how,
        )

    def join_asof(self, view: FeatureViewWrapper, on_left: list[str], on_right: list[str]) -> BatchDataSource:
        from aligned.data_source.batch_data_source import join_asof_source
        from aligned.schemas.model import ModelSource

        compiled_model = self.compile()
        compiled_view = self.as_view()

        if compiled_view is None:
            raise ValueError(f"Model {compiled_model.name} is not compiled as a view")

        source = ModelSource(compiled_model, compiled_view)

        return join_asof_source(
            source,
            view=view,
            left_on=on_left,
            right_on=on_right,
            left_request=compiled_view.request_all.needed_requests[0],
        )


def resolve_dataset_store(dataset_store: DatasetStore | StorageFileReference) -> DatasetStore:
    from aligned.schemas.folder import DatasetStore, JsonDatasetStore

    if isinstance(dataset_store, DatasetStore):
        return dataset_store

    return JsonDatasetStore(dataset_store)


@dataclass
class FeatureInputVersions:

    default_version: str
    versions: dict[str, list[FeatureReferencable]]

    def compile(self) -> FeatureVersionSchema:
        return FeatureVersionSchema(
            default_version=self.default_version,
            versions={
                version: [feature.feature_referance() for feature in features]
                for version, features in self.versions.items()
            },
        )


def model_contract(
    name: str,
    features: list[FeatureReferencable] | FeatureInputVersions,
    contacts: list[str] | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    prediction_source: BatchDataSource | None = None,
    prediction_stream: StreamDataSource | None = None,
    application_source: BatchDataSource | None = None,
    dataset_store: DatasetStore | StorageFileReference | None = None,
    exposed_at_url: str | None = None,
    acceptable_freshness: timedelta | None = None,
    unacceptable_freshness: timedelta | None = None,
) -> Callable[[Type[T]], ModelContractWrapper[T]]:
    def decorator(cls: Type[T]) -> ModelContractWrapper[T]:

        if isinstance(features, FeatureInputVersions):
            input_features = features
        else:
            input_features = FeatureInputVersions(default_version='default', versions={'default': features})

        metadata = ModelMetadata(
            name,
            input_features,
            contacts=contacts,
            tags=tags,
            description=description,
            prediction_source=prediction_source,
            prediction_stream=prediction_stream,
            application_source=application_source,
            dataset_store=resolve_dataset_store(dataset_store) if dataset_store else None,
            exposed_at_url=exposed_at_url,
            acceptable_freshness=acceptable_freshness,
            unacceptable_freshness=unacceptable_freshness,
        )
        return ModelContractWrapper(metadata, cls)

    return decorator


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
        recommendation_targets=set(),
        acceptable_freshness=metadata.acceptable_freshness,
        unacceptable_freshness=metadata.unacceptable_freshness,
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

        elif isinstance(feature, ModelContractWrapper):
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
            assert feature.target._name in classification_targets, 'Target must be a classification target.'

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
        elif isinstance(feature, RecommendationTarget):
            inference_view.recommendation_targets.add(feature.compile())
        elif isinstance(feature, Entity):
            inference_view.entities.add(feature.feature())
        elif isinstance(feature, FeatureFactory):
            inference_view.features.add(feature.feature())

    # Needs to run after the feature views have compiled
    features = metadata.features.compile()

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
        inference_view.features.update({target.feature for target in inference_view.classification_targets})

    return ModelSchema(
        name=metadata.name,
        features=features,
        predictions_view=inference_view,
        contacts=metadata.contacts,
        tags=metadata.tags,
        description=metadata.description,
        dataset_store=metadata.dataset_store,
        exposed_at_url=metadata.exposed_at_url,
    )
