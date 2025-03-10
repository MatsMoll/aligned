from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from aligned.schemas.codable import Codable
from aligned.schemas.feature import (
    Constraint,
    Feature,
    FeatureLocation,
    FeatureReference,
    FeatureType,
)
from aligned.schemas.transformation import Transformation


class DerivedFeature(Feature):
    depending_on: set[FeatureReference]
    transformation: Transformation
    loads_feature: FeatureReference | None = None
    depth: int = 1

    def __init__(
        self,
        name: str,
        dtype: FeatureType,
        depending_on: set[FeatureReference],
        transformation: Transformation,
        depth: int,
        description: str | None = None,
        tags: list[str] | None = None,
        constraints: set[Constraint] | None = None,
        loads_feature: FeatureReference | None = None,
    ):
        self.name = name
        self.tags = tags
        self.depth = depth
        self.dtype = dtype
        self.description = description
        self.constraints = constraints
        self.depending_on = depending_on
        self.transformation = transformation
        self.loads_feature = loads_feature
        self.default_value = None

    def __pre_serialize__(self) -> DerivedFeature:
        from aligned.schemas.transformation import SupportedTransformations

        for feature in self.depending_on:
            assert isinstance(feature, FeatureReference)

        assert isinstance(self.transformation, Transformation)
        assert self.transformation.name in SupportedTransformations.shared().types

        return self

    @property
    def depending_on_names(self) -> list[str]:
        return [feature.name for feature in self.depending_on]

    @property
    def depending_on_views(self) -> set[FeatureLocation]:
        return {feature.location for feature in self.depending_on}

    @property
    def feature(self) -> Feature:
        return Feature(
            name=self.name,
            dtype=self.dtype,
            description=self.description,
            tags=self.tags,
            constraints=self.constraints,
        )


@dataclass
class AggregationTimeWindow(Codable):
    time_window: timedelta
    time_column: FeatureReference

    every_interval: timedelta | None = field(default=None)
    offset_interval: timedelta | None = field(default=None)

    def __hash__(self) -> int:
        return self.time_window.__hash__()


@dataclass
class AggregateOver(Codable):
    group_by: list[FeatureReference]
    window: AggregationTimeWindow | None = field(default=None)
    condition: DerivedFeature | None = field(default=None)

    @property
    def group_by_names(self) -> list[str]:
        return [feature.name for feature in self.group_by]

    def __hash__(self) -> int:
        if self.window:
            return self.window.__hash__()

        name = ""
        for feature in self.group_by:
            name += feature.name
        return name.__hash__()


@dataclass
class AggregatedFeature(Codable):
    derived_feature: DerivedFeature
    aggregate_over: AggregateOver

    def __hash__(self) -> int:
        return self.derived_feature.name.__hash__()

    @property
    def depending_on(self) -> set[FeatureReference]:
        return self.derived_feature.depending_on

    @property
    def depending_on_names(self) -> list[str]:
        return [feature.name for feature in self.depending_on]

    @property
    def depending_on_views(self) -> set[FeatureLocation]:
        return {feature.location for feature in self.depending_on}

    @property
    def feature(self) -> Feature:
        return self.derived_feature.feature

    @property
    def name(self) -> str:
        return self.derived_feature.name
