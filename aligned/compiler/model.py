from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Type, TypeVar, Generic, TYPE_CHECKING
from datetime import timedelta

from uuid import uuid4

from aligned.compiler.feature_factory import (
    CanBeClassificationLabel,
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
from aligned.exposed_model.interface import ExposedModel
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import ConvertableToRetrivalJob, PredictionJob, RetrivalJob
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReference, FeatureType
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
    from aligned.feature_store import ModelFeatureStore

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

    output_source: BatchDataSource | None = field(default=None)
    output_stream: StreamDataSource | None = field(default=None)

    application_source: BatchDataSource | None = field(default=None)

    acceptable_freshness: timedelta | None = field(default=None)
    unacceptable_freshness: timedelta | None = field(default=None)

    exposed_at_url: str | None = field(default=None)
    exposed_model: ExposedModel | None = field(default=None)

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

    def query(
        self, needed_views: list[FeatureViewWrapper | ModelContractWrapper] | None = None
    ) -> ModelFeatureStore:
        from aligned import ContractStore

        store = ContractStore.empty()
        store.add_model(self)

        for needed_data in needed_views or []:
            if isinstance(needed_data, ModelContractWrapper):
                store.add_compiled_model(needed_data.compile())
            else:
                store.add_compiled_view(needed_data.compile())

        return store.model(self.metadata.name)

    def predict_over(
        self,
        values: ConvertableToRetrivalJob | RetrivalJob,
        needed_views: list[FeatureViewWrapper | ModelContractWrapper] | None = None,
    ) -> PredictionJob:
        from aligned.retrival_job import RetrivalJob

        model = self.compile()

        if not model.exposed_model:
            raise ValueError(f"Model {model.name} does not have an `exposed_model` to use for predictions.")

        if not isinstance(values, RetrivalJob):
            features = {feat.as_feature() for feat in model.features.default_features}
            request = RetrivalRequest(
                name='default',
                location=FeatureLocation.model(model.name),
                entities=set(),
                features=features,
                derived_features=set(),
            )
            values = RetrivalJob.from_convertable(values, request)

        return self.query(needed_views).predict_over(values)

    def as_view(self) -> CompiledFeatureView | None:

        compiled = self.compile()
        view = compiled.predictions_view

        return view.as_view(self.metadata.name)

    def filter(
        self, name: str, where: Callable[[T], Bool], application_source: BatchDataSource | None = None
    ) -> ModelContractWrapper[T]:
        from aligned.data_source.batch_data_source import FilteredDataSource

        meta = self.metadata
        meta.name = name

        condition = where(self.__call__())

        main_source = meta.output_source
        if not main_source:
            raise ValueError(
                f'Model: {self.metadata.name} needs a `prediction_source` to use `filter`, got None.'
            )

        if not condition._name:
            condition._name = str(uuid4())
            condition._location = FeatureLocation.model(name)

        if condition.transformation:
            meta.output_source = FilteredDataSource(main_source, condition.compile())
        else:
            meta.output_source = FilteredDataSource(main_source, condition.feature())

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
                version: [feature.feature_reference() for feature in features]
                for version, features in self.versions.items()
            },
        )


def model_contract(
    input_features: list[FeatureReferencable | FeatureViewWrapper | ModelContractWrapper]
    | FeatureInputVersions,
    name: str | None = None,
    contacts: list[str] | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    output_source: BatchDataSource | None = None,
    output_stream: StreamDataSource | None = None,
    application_source: BatchDataSource | None = None,
    dataset_store: DatasetStore | StorageFileReference | None = None,
    exposed_at_url: str | None = None,
    exposed_model: ExposedModel | None = None,
    acceptable_freshness: timedelta | None = None,
    unacceptable_freshness: timedelta | None = None,
) -> Callable[[Type[T]], ModelContractWrapper[T]]:
    def decorator(cls: Type[T]) -> ModelContractWrapper[T]:

        if isinstance(input_features, FeatureInputVersions):
            features_versions = input_features
        else:
            unwrapped_input_features: list[FeatureReferencable] = []

            for feature in input_features:
                if isinstance(feature, FeatureViewWrapper):
                    compiled_view = feature.compile()
                    request = compiled_view.request_all
                    features = [
                        feat.as_reference(FeatureLocation.feature_view(compiled_view.name))
                        for feat in request.request_result.features
                    ]
                    unwrapped_input_features.extend(features)
                elif isinstance(feature, ModelContractWrapper):
                    compiled_model = feature.compile()
                    request = compiled_model.predictions_view.request('')
                    features = [
                        feat.as_reference(FeatureLocation.model(compiled_model.name))
                        for feat in request.request_result.features
                    ]
                    unwrapped_input_features.extend(features)
                else:
                    unwrapped_input_features.append(feature)

            features_versions = FeatureInputVersions(
                default_version='default', versions={'default': unwrapped_input_features}
            )

        used_name = name or str(cls.__name__).lower()

        used_description = None
        if description:
            used_description = description
        elif cls.__doc__:
            used_description = str(cls.__doc__)

        used_exposed_at_url = exposed_at_url
        if exposed_model:
            used_exposed_at_url = exposed_model.exposed_at_url or exposed_at_url

        metadata = ModelMetadata(
            used_name,
            features_versions,
            contacts=contacts,
            tags=tags,
            description=used_description,
            output_source=output_source,
            output_stream=output_stream,
            application_source=application_source,
            dataset_store=resolve_dataset_store(dataset_store) if dataset_store else None,
            exposed_at_url=used_exposed_at_url,
            exposed_model=exposed_model,
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
        source=metadata.output_source,
        application_source=metadata.application_source,
        stream_source=metadata.output_stream,
        classification_targets=set(),
        regression_targets=set(),
        recommendation_targets=set(),
        acceptable_freshness=metadata.acceptable_freshness,
        unacceptable_freshness=metadata.unacceptable_freshness,
    )
    probability_features: dict[str, set[TargetProbability]] = {}
    hidden_features = 0

    classification_targets: dict[str, ClassificationTargetSchema] = {}
    regression_targets: dict[str, RegressionTargetSchema] = {}

    for var_name in var_names:
        feature = getattr(model, var_name)
        if isinstance(feature, FeatureFactory):
            assert feature._name
            feature._location = FeatureLocation.model(metadata.name)

        if isinstance(feature, ModelVersion):
            inference_view.model_version_column = feature.feature()

        if isinstance(feature, FeatureView):
            compiled = feature.compile()
            inference_view.entities.update(compiled.entities)

        elif isinstance(feature, ModelContractWrapper):
            compiled = feature.compile()
            inference_view.entities.update(compiled.predictions_view.entities)

        elif (
            isinstance(feature, CanBeClassificationLabel)
            and (target_feature := feature.compile_classification_target()) is not None
        ):
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
            if feature.transformation:
                # Adding features that is not stored in the view
                # e.g:
                # class SomeView(FeatureView):
                #     ...
                #     x, y = Bool(), Bool()
                #     z = (x & y) | x
                #
                # Here will (x & y)'s result be a 'hidden' feature
                feature_deps = [(feat.depth(), feat) for feat in feature.feature_dependencies()]

                # Sorting by key in order to instanciate the "core" features first
                # And then making it possible for other features to reference them
                def sort_key(x: tuple[int, FeatureFactory]) -> int:
                    return x[0]

                for depth, feature_dep in sorted(feature_deps, key=sort_key):

                    if not feature_dep._location:
                        feature_dep._location = FeatureLocation.feature_view(metadata.name)
                    elif feature_dep._location.name != metadata.name:
                        continue

                    if feature_dep._name:
                        feat_dep = feature_dep.feature()
                        if feat_dep in inference_view.features or feat_dep in inference_view.entities:
                            continue

                    if depth == 0:
                        if not feature_dep._name:
                            feature_dep._name = var_name

                        feat_dep = feature_dep.feature()
                        inference_view.features.add(feat_dep)
                        continue

                    if not feature_dep._name:
                        feature_dep._name = str(hidden_features)
                        hidden_features += 1

                    feature_graph = feature_dep.compile()  # Should decide on which payload to send
                    if feature_graph in inference_view.derived_features:
                        continue

                    inference_view.derived_features.add(feature_dep.compile())

                inference_view.derived_features.add(feature.compile())
            else:
                inference_view.features.add(feature.feature())

        if isinstance(feature, Bool) and feature._is_shadow_model_flag:
            inference_view.is_shadow_model_flag = feature.feature()

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
                FeatureReference(feat, FeatureLocation.model(metadata.name), dtype=FeatureType.float())
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
        exposed_model=metadata.exposed_model,
    )
