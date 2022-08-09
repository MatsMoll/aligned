from abc import ABC, abstractproperty
from typing import Callable, TypeVar

from aladdin.feature_types import Entity, EventTimestamp, FeatureFactory, TransformationFactory
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.feature_view.feature_view_metadata import FeatureViewMetadata
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest

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
        metadata = cls().metadata
        var_names = [name for name in cls().__dir__() if not name.startswith('_')]

        view = CompiledFeatureView(
            name=metadata.name,
            description=metadata.description,
            tags=metadata.tags,
            batch_data_source=metadata.batch_source,
            entities=set(),
            features=set(),
            derived_features=set(),
            event_timestamp=None,
            stream_data_source=metadata.stream_source,
        )

        def add_sub_features(feature: TransformationFactory) -> None:

            for sub_feature in feature.using_features:
                if not isinstance(sub_feature, TransformationFactory) or sub_feature._name:
                    continue

                add_sub_features(sub_feature)

                sub_feature._feature_view = metadata.name
                view.derived_features.add(
                    sub_feature.derived_feature(
                        name=sub_feature.name,
                        sources=[
                            (
                                metadata,
                                view.request_for({sub_feature.name}).needed_requests[0],
                            )
                        ],
                    )
                )

        for var_name in var_names:
            feature = getattr(cls, var_name)

            if isinstance(feature, TransformationFactory):
                feature._name = var_name
                feature._feature_view = metadata.name

                add_sub_features(feature)

                view.derived_features.add(
                    feature.derived_feature(
                        name=feature.name,
                        sources=[
                            (
                                metadata,
                                view.request_for({feature.name}).needed_requests[0],
                            )
                        ],
                    )
                )

                continue
            elif not isinstance(feature, FeatureFactory):
                continue

            feature._feature_view = metadata.name
            if isinstance(feature, Entity):
                view.entities.add(feature.feature(var_name))
            elif isinstance(feature, EventTimestamp):
                if view.event_timestamp is not None:
                    raise Exception(
                        'Can only have one EventTimestamp for each'
                        ' FeatureViewDefinition. Check that this is the case for'
                        f' {cls.__name__}'
                    )
                view.features.add(feature.feature(var_name))
                view.event_timestamp = feature.event_timestamp_feature(var_name)
            else:
                view.features.add(feature.feature(var_name))

        if not view.entities:
            raise ValueError(f'FeatureView {metadata.name} must contain at least one Entity')

        return view

    @classmethod
    def select(cls: type[FVType], features: Callable[[type[FVType]], list[FeatureFactory]]) -> FeatureRequest:
        view: CompiledFeatureView = cls.compile()  # type: ignore
        names = features(cls)
        return view.request_for({feature.name for feature in names if feature.name})

    @classmethod
    def select_all(cls: type[FVType]) -> FeatureRequest:
        return cls.compile().request_all  # type: ignore
