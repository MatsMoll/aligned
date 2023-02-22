from dataclasses import dataclass
from datetime import timedelta

from aligned.compiler.feature_factory import (
    AggregationTransformationFactory,
    FeatureFactory,
    FeatureReferance,
    String,
    TransformationFactory,
)
from aligned.schemas.transformation import Transformation


@dataclass
class ConcatStringsAggrigationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: String
    group_by: list[FeatureReferance]
    separator: str | None = None
    time_window: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import ConcatStringAggregation

        return ConcatStringAggregation(
            key=self.feature.feature_referance().name,
            group_keys=[feature.name for feature in self.group_by],
            separator=self.separator,
        )

    def with_group_by(self, values: list[FeatureReferance]) -> TransformationFactory:
        self.group_by = values
        return self


@dataclass
class SumAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    group_by: list[FeatureReferance]
    time_window: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import SumAggregation

        return SumAggregation(
            key=self.feature.feature_referance().name,
            group_keys=[feature.name for feature in self.group_by],
        )

    def with_group_by(self, values: list[FeatureReferance]) -> TransformationFactory:
        self.group_by = values
        return self
