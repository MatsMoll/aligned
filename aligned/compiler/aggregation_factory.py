import polars as pl
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

from aligned.compiler.feature_factory import (
    AggregationTransformationFactory,
    FeatureFactory,
    FeatureReferance,
    String,
    TransformationFactory,
)
from aligned.schemas.derivied_feature import AggregateOver, AggregationTimeWindow, DerivedFeature
from aligned.schemas.transformation import Transformation


def aggregate_over(
    group_by: list[FeatureReferance],
    time_column: FeatureReferance | None,
    time_window: timedelta | None,
    every_interval: timedelta | None,
    offset_interval: timedelta | None,
    condition: DerivedFeature | None,
) -> AggregateOver:
    if not time_window:
        return AggregateOver(group_by)

    if not time_column:
        raise ValueError(
            f'Aggregation {group_by} over {time_column} have a time window, but no event timestamp to use'
        )

    return AggregateOver(
        group_by,
        AggregationTimeWindow(time_window, time_column, every_interval, offset_interval=offset_interval),
        condition=condition,
    )


@dataclass
class ConcatStringsAggrigationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: String
    separator: str | None = None
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import ConcatStringAggregation

        return ConcatStringAggregation(
            key=self.feature.feature_referance().name,
            separator=self.separator or '',
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class SumAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import SumAggregation

        return SumAggregation(
            key=self.feature.feature_referance().name,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class MeanAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import MeanAggregation

        return MeanAggregation(
            key=self.feature.feature_referance().name,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class MinAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import MinAggregation

        return MinAggregation(
            key=self.feature.feature_referance().name,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class MaxAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import MaxAggregation

        return MaxAggregation(
            key=self.feature.feature_referance().name,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class CountAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import CountAggregation

        return CountAggregation(
            key=self.feature.feature_referance().name,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class CountDistinctAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import CountDistinctAggregation

        return CountDistinctAggregation(
            key=self.feature.feature_referance().name,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class StdAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import StdAggregation

        return StdAggregation(
            key=self.feature.feature_referance().name,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class VarianceAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import VarianceAggregation

        return VarianceAggregation(
            key=self.feature.feature_referance().name,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class MedianAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import MedianAggregation

        return MedianAggregation(
            key=self.feature.feature_referance().name,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class PercentileAggregationFactory(TransformationFactory, AggregationTransformationFactory):

    feature: FeatureFactory
    percentile: float
    time_window: timedelta | None = None
    offset_interval: timedelta | None = None
    every_interval: timedelta | None = None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import PercentileAggregation

        return PercentileAggregation(
            key=self.feature.feature_referance().name,
            percentile=self.percentile,
        )

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(
            group_by, time_column, self.time_window, self.every_interval, self.offset_interval, None
        )


@dataclass
class PolarsTransformationFactoryAggregation(TransformationFactory, AggregationTransformationFactory):

    dtype: FeatureFactory
    method: pl.Expr | Callable[[pl.LazyFrame, pl.Expr], pl.LazyFrame]
    _using_features: list[FeatureFactory]

    @property
    def using_features(self) -> list[FeatureFactory]:
        return self._using_features

    def aggregate_over(
        self, group_by: list[FeatureReferance], time_column: FeatureReferance | None
    ) -> AggregateOver:
        return aggregate_over(group_by, time_column, None, None, None, None)

    def compile(self) -> Transformation:
        import inspect
        import types

        import dill

        from aligned.schemas.transformation import PolarsFunctionTransformation, PolarsLambdaTransformation

        if isinstance(self.method, pl.Expr):
            method = lambda df, alias: self.method  # type: ignore
            code = ''
            return PolarsLambdaTransformation(method=dill.dumps(method), code=code, dtype=self.dtype.dtype)
        else:
            code = inspect.getsource(self.method)

        if isinstance(self.method, types.LambdaType) and self.method.__name__ == '<lambda>':
            return PolarsLambdaTransformation(
                method=dill.dumps(self.method), code=code.strip(), dtype=self.dtype.dtype
            )
        else:
            return PolarsFunctionTransformation(
                code=code,
                function_name=dill.source.getname(self.method),
                dtype=self.dtype.dtype,
            )
