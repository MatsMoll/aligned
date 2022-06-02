from abc import ABC, abstractproperty
from typing import Callable, TypeVar

from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import EventTimestamp as EventTimestampFeature
from aladdin.feature import Feature
from aladdin.feature_types import Entity, EventTimestamp, FeatureFactory, TransformationFactory
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.feature_view.feature_view_metadata import FeatureViewMetadata
from aladdin.request.retrival_request import RetrivalRequest

# Enables code compleation in the select method
FVType = TypeVar('FVType')


class FeatureSelectable:
    @classmethod
    def select(
        cls: type[FVType], features: Callable[[type[FVType]], list[FeatureFactory]]
    ) -> RetrivalRequest:
        pass

    @classmethod
    def select_all(cls: type[FVType]) -> RetrivalRequest:
        pass


class FeatureView(ABC, FeatureSelectable):
    """
    A collection of features, and a way to combine them.

    This should contain the core features, and might contain derived features (aka. transformations).
    """

    @abstractproperty
    def metadata(self) -> FeatureViewMetadata:
        pass

    @classmethod
    def compile(cls) -> CompiledFeatureView:
        features: list[Feature] = []
        metadata = cls().metadata
        event_timestamp: EventTimestampFeature | None = None
        entities: list[Feature] = []
        transformations: list[DerivedFeature] = []
        var_names = [name for name in cls().__dir__() if not name.startswith('_')]
        for var_name in var_names:
            feature = getattr(cls, var_name)

            if isinstance(feature, TransformationFactory):
                if event_timestamp is None:
                    raise Exception(f'Set EventTimestamp above the feature {var_name} in' f' {cls.__name__}')

                feature._name = var_name
                feature._feature_view = metadata.name
                tran = feature.transformation(
                    [
                        (
                            metadata,
                            RetrivalRequest(
                                feature_view_name=metadata.name,
                                entities=set(entities),
                                features=set(features),
                                derived_features=set(transformations),
                                event_timestamp=event_timestamp,
                            ),
                        )
                    ]
                )
                transformations.append(
                    DerivedFeature(
                        name=var_name,
                        dtype=feature.feature._dtype,
                        depending_on={feature.feature_referance() for feature in feature.using_features},
                        transformation=tran,
                    )
                )
                continue
            elif not isinstance(feature, FeatureFactory):
                continue

            feature._feature_view = metadata.name
            if isinstance(feature, Entity):
                entities.append(feature.feature(var_name))
            elif isinstance(feature, EventTimestamp):
                if event_timestamp is not None:
                    raise Exception(
                        'Can only have one EventTimestamp for each'
                        ' FeatureViewDefinition. Check that this is the case for'
                        f' {cls.__name__}'
                    )
                features.append(feature.feature(var_name))
                event_timestamp = feature.event_timestamp_feature(var_name)
            else:
                features.append(feature.feature(var_name))

        if not entities:
            raise ValueError(f'FeatureView {metadata.name} must contain at least one Entity')

        if event_timestamp is None:
            raise Exception(f'A EventTimestamp is needed for {cls.__name__}')

        return CompiledFeatureView(
            name=metadata.name,
            description=metadata.description,
            tags=metadata.tags,
            batch_data_source=metadata.batch_source,
            # stream_data_source=metadata.stream_data_source,
            entities=set(entities),
            features=set(features),
            derived_features=set(transformations),
            event_timestamp=event_timestamp,
        )

    @classmethod
    def select(
        cls: type[FVType], features: Callable[[type[FVType]], list[FeatureFactory]]
    ) -> RetrivalRequest:
        view: CompiledFeatureView = cls.compile()  # type: ignore
        names = features(cls)
        return view.request_for({feature.name for feature in names if feature.name})

    @classmethod
    def select_all(cls: type[FVType]) -> RetrivalRequest:
        return cls.compile().request_all  # type: ignore
