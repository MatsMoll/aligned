from abc import ABC, abstractproperty
from typing import Callable, TypeVar

from aladdin.compiler.feature_factory import Entity, EventTimestamp, FeatureFactory
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.feature_view.feature_view_metadata import FeatureViewMetadata
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest
from aladdin.schemas.feature import Feature

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
    def select(cls: type[FVType], features: Callable[[type[FVType]], list[FeatureFactory]]) -> FeatureRequest:
        view: CompiledFeatureView = cls.compile_graph_only()  # type: ignore
        names = features(cls)
        return view.request_for({feature.name for feature in names if feature.name})

    @classmethod
    def select_all(cls: type[FVType]) -> FeatureRequest:
        return cls.compile_graph_only().request_all  # type: ignore

    @classmethod
    async def compile(cls) -> CompiledFeatureView:
        from aladdin.compiler.feature_factory import FeatureFactory

        # Used to deterministicly init names for hidden features
        hidden_features = 0

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

        for var_name in var_names:
            feature = getattr(cls, var_name)

            if not isinstance(feature, FeatureFactory):
                continue

            feature._name = var_name
            feature._feature_view = metadata.name
            compiled_feature = await feature.feature()

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

                    if feature_dep._name:
                        continue

                    if depth == 0:
                        feature_dep._name = var_name
                        view.features.add(compiled_feature)
                        continue

                    feature_dep._name = str(hidden_features)
                    hidden_features += 1
                    view.derived_features.add(
                        await feature_dep.compile([view])  # Should decide on which payload to send
                    )

                view.derived_features.add(
                    await feature.compile([view])  # Should decide on which payload to send
                )

            elif isinstance(feature, Entity):
                view.entities.add(compiled_feature)
            elif isinstance(feature, EventTimestamp):
                if view.event_timestamp is not None:
                    raise Exception(
                        'Can only have one EventTimestamp for each'
                        ' FeatureViewDefinition. Check that this is the case for'
                        f' {cls.__name__}'
                    )
                view.features.add(compiled_feature)
                view.event_timestamp = feature.event_timestamp()
            else:
                view.features.add(compiled_feature)

        if not view.entities:
            raise ValueError(f'FeatureView {metadata.name} must contain at least one Entity')

        return view

    @classmethod
    def compile_graph_only(cls) -> CompiledFeatureView:
        from aladdin.compiler.feature_factory import FeatureFactory

        # Used to deterministicly init names for hidden features
        hidden_features = 0

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

        for var_name in var_names:
            feature = getattr(cls, var_name)

            if not isinstance(feature, FeatureFactory):
                continue

            feature._name = var_name
            feature._feature_view = metadata.name
            compiled_feature = Feature(name=feature._name, dtype=feature.dtype)

            if feature.transformation:
                feature_deps = [(feat.depth(), feat) for feat in feature.feature_dependencies()]

                # Sorting by key in order to instanciate the "core" features first
                # And then making it possible for other features to reference them
                def sort_key(x: tuple[int, FeatureFactory]) -> int:
                    return x[0]

                for _, feature_dep in sorted(feature_deps, key=sort_key):

                    if feature_dep._name:
                        continue

                    feature_dep._name = str(hidden_features)
                    hidden_features += 1
                    view.derived_features.add(
                        feature_dep.compile_graph_only()  # Should decide on which payload to send
                    )

                view.derived_features.add(
                    feature.compile_graph_only()  # Should decide on which payload to send
                )

            elif isinstance(feature, Entity):
                view.entities.add(compiled_feature)
            elif isinstance(feature, EventTimestamp):
                if view.event_timestamp is not None:
                    raise Exception(
                        'Can only have one EventTimestamp for each'
                        ' FeatureViewDefinition. Check that this is the case for'
                        f' {cls.__name__}'
                    )
                view.features.add(compiled_feature)
                view.event_timestamp = feature.event_timestamp()
            else:
                view.features.add(compiled_feature)

        if not view.entities:
            raise ValueError(f'FeatureView {metadata.name} must contain at least one Entity')

        return view
