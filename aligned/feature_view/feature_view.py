from __future__ import annotations

from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from aligned.compiler.feature_factory import (
    AggregationTransformationFactory,
    Embedding,
    Entity,
    EventTimestamp,
)
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.schemas.derivied_feature import AggregatedFeature, AggregateOver, AggregationTimeWindow
from aligned.schemas.feature import FeatureLocation, FeatureReferance
from aligned.schemas.feature_view import CompiledFeatureView

if TYPE_CHECKING:
    from aligned.feature_store import FeatureViewStore

# Enables code compleation in the select method
FVType = TypeVar('FVType')


@dataclass
class FeatureViewMetadata:
    name: str
    description: str
    batch_source: BatchDataSource
    stream_source: StreamDataSource | None = field(default=None)
    contacts: list[str] | None = field(default=None)
    tags: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_compiled(view: CompiledFeatureView) -> FeatureViewMetadata:
        return FeatureViewMetadata(
            name=view.name,
            description=view.description,
            tags=view.tags,
            batch_source=view.batch_data_source,
            stream_source=view.stream_data_source,
        )


class FeatureView(ABC):
    """
    A collection of features, and a way to combine them.

    This should contain the core features, and might contain derived features (aka. transformations).
    """

    @abstractproperty
    def metadata(self) -> FeatureViewMetadata:
        pass

    @staticmethod
    def metadata_with(
        name: str,
        description: str,
        batch_source: BatchDataSource,
        stream_source: StreamDataSource | None = None,
        contacts: list[str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> FeatureViewMetadata:
        from aligned import HttpStreamSource

        return FeatureViewMetadata(
            name,
            description,
            batch_source,
            stream_source or HttpStreamSource(name),
            contacts=contacts,
            tags=tags or {},
        )

    @classmethod
    def compile(cls) -> CompiledFeatureView:
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
            aggregated_features=set(),
            event_timestamp=None,
            stream_data_source=metadata.stream_source,
            indexes=[],
        )
        aggregations: list[FeatureFactory] = []

        for var_name in var_names:
            feature = getattr(cls, var_name)

            if not isinstance(feature, FeatureFactory):
                continue

            feature._name = var_name
            feature._location = FeatureLocation.feature_view(metadata.name)
            compiled_feature = feature.feature()

            if isinstance(feature, Embedding) and feature.indexes:
                view.indexes.extend(
                    [
                        index.compile(feature._location, compiled_feature, view.entities)
                        for index in feature.indexes
                    ]
                )

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

                    if feature_dep._name:
                        feat_dep = feature_dep.feature()
                        if feat_dep in view.features or feat_dep in view.entities:
                            continue

                    if depth == 0:
                        # The raw value and the transformed have the same name
                        feature_dep._name = var_name
                        feat_dep = feature_dep.feature()
                        view.features.add(feat_dep)
                        continue

                    if not feature_dep._name:
                        feature_dep._name = str(hidden_features)
                        hidden_features += 1

                    if isinstance(feature_dep.transformation, AggregationTransformationFactory):
                        aggregations.append(feature_dep)
                    else:
                        feature_graph = feature_dep.compile()  # Should decide on which payload to send
                        if feature_graph in view.derived_features:
                            continue

                        view.derived_features.add(feature_dep.compile())

                if isinstance(feature.transformation, AggregationTransformationFactory):
                    aggregations.append(feature)
                else:
                    view.derived_features.add(feature.compile())  # Should decide on which payload to send

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

        aggregation_group_by = [
            FeatureReferance(entity.name, FeatureLocation.feature_view(view.name), entity.dtype)
            for entity in view.entities
        ]

        for aggr in aggregations:
            agg_trans = aggr.transformation
            if not isinstance(agg_trans, AggregationTransformationFactory):
                continue

            if view.event_timestamp is None and agg_trans.time_window:
                raise ValueError(f'FeatureView {metadata.name} must contain an EventTimestamp')

            time_window: AggregationTimeWindow | None = None
            if agg_trans.time_window:

                timestamp_ref = FeatureReferance(
                    view.event_timestamp.name,
                    FeatureLocation.feature_view(view.name),
                    dtype=view.event_timestamp.dtype,
                )
                time_window = AggregationTimeWindow(agg_trans.time_window, timestamp_ref)

            aggr.transformation = agg_trans.with_group_by(aggregation_group_by)
            config = AggregateOver(aggregation_group_by, window=time_window, condition=None)
            feature = aggr.compile()
            feat = AggregatedFeature(
                derived_feature=feature,
                aggregate_over=config,
            )
            view.aggregated_features.add(feat)

        return view

    @classmethod
    def query(cls) -> FeatureViewStore:
        from aligned import FeatureStore

        self = cls()
        store = FeatureStore.experimental()
        store.add_feature_view(self)
        return store.feature_view(self.metadata.name)
