from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from aligned.compiler.feature_factory import Entity, EventTimestamp, FeatureFactory
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.schemas.feature import Feature
from aligned.schemas.feature_view import CompiledFeatureView

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


@dataclass
class FeatureViewMetadata:
    name: str
    description: str
    batch_source: BatchDataSource
    stream_source: StreamDataSource | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_compiled(view: CompiledFeatureView) -> 'FeatureViewMetadata':
        return FeatureViewMetadata(
            name=view.name,
            description=view.description,
            tags=view.tags,
            batch_source=view.batch_data_source,
            stream_source=None,
        )


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
        from aligned.compiler.feature_factory import FeatureFactory

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

                    if not feature_dep._feature_view:
                        feature_dep._feature_view = metadata.name

                    if feature_dep._name:
                        feat_dep = await feature_dep.feature()
                        if feat_dep in view.features or feat_dep in view.entities:
                            continue

                    if depth == 0:
                        feature_dep._name = var_name
                        feat_dep = await feature_dep.feature()
                        view.features.add(feat_dep)
                        continue

                    if not feature_dep._name:
                        feature_dep._name = str(hidden_features)
                        hidden_features += 1

                    feature_graph = feature_dep.compile_graph_only()  # Should decide on which payload to send
                    if feature_graph in view.derived_features:
                        continue

                    view.derived_features.add(await feature_dep.compile([view]))

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
        """Compiles a view with all its metadata,
        However it is not containing the correct compiled transofmrations in all cases
        As it will nto compute the different artefacts.

        This is a flaw in the current system and should potentially be fixed by creating artefacts
        based on data models and not views. As that is more data set spesifics.
        It would therefore not need two different compile methods
        """
        from aligned.compiler.feature_factory import FeatureFactory

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

                for depth, feature_dep in sorted(feature_deps, key=sort_key):

                    if not feature_dep._feature_view:
                        feature_dep._feature_view = metadata.name

                    if feature_dep._name:
                        feat_dep = Feature(feature_dep._name, dtype=feature_dep.dtype)
                        if feat_dep in view.features or feat_dep in view.entities:
                            continue

                    if depth == 0:
                        feature_dep._name = var_name
                        feat_dep = Feature(var_name, dtype=feature_dep.dtype)
                        view.features.add(feat_dep)
                        continue

                    if not feature_dep._name:
                        feature_dep._name = str(hidden_features)
                        hidden_features += 1

                    feature_graph = feature_dep.compile_graph_only()  # Should decide on which payload to send
                    if feature_graph in view.derived_features:
                        continue

                    view.derived_features.add(
                        feature_dep.compile_graph_only()  # Should decide on which payload to send
                    )
                view.derived_features.add(feature.compile_graph_only())
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
