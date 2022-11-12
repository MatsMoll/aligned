import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

from pandas import DataFrame, Series

from aligned.compiler.feature_factory import FeatureFactory, Transformation, TransformationFactory
from aligned.enricher import StatisticEricher, TimespanSelector
from aligned.exceptions import InvalidStandardScalerArtefact
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.schemas.transformation import StandardScalingTransformation


@dataclass
class EqualsFactory(TransformationFactory):

    left: FeatureFactory
    right: FeatureFactory | Any

    @property
    def using_features(self) -> list[FeatureFactory]:
        if isinstance(self.right, FeatureFactory):
            return [self.left, self.right]
        else:
            return [self.left]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Equals

        if isinstance(self.right, FeatureFactory):
            raise NotImplementedError()
        else:
            return Equals(self.left.name, self.right)


@dataclass
class RatioFactory(TransformationFactory):

    numerator: FeatureFactory
    denumerator: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.numerator, self.denumerator]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Ratio

        return Ratio(self.numerator.name, self.denumerator.name)


@dataclass
class OrdinalFactory(TransformationFactory):

    orders: list[str]
    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Ordinal

        return Ordinal(self.feature.name, self.orders)


@dataclass
class ContainsFactory(TransformationFactory):

    text: str
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Contains as ContainsTransformation

        return ContainsTransformation(self.in_feature.name, self.text)


@dataclass
class NotEqualsFactory(TransformationFactory):

    value: Any
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import NotEquals as NotEqualsTransformation

        return NotEqualsTransformation(self.in_feature.name, self.value)


@dataclass
class GreaterThenFactory(TransformationFactory):

    left_feature: FeatureFactory
    right: Any

    @property
    def using_features(self) -> list[FeatureFactory]:
        if isinstance(self.right, FeatureFactory):
            return [self.left_feature, self.right]
        else:
            return [self.left_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import GreaterThen, GreaterThenValue

        if isinstance(self.right, FeatureFactory):
            return GreaterThen(self.left_feature.name, self.right.name)
        else:
            return GreaterThenValue(self.left_feature.name, self.right)


@dataclass
class GreaterThenOrEqualFactory(TransformationFactory):

    value: Any
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import GreaterThenOrEqual as GTETransformation

        return GTETransformation(self.in_feature.name, self.value)


@dataclass
class LowerThenFactory(TransformationFactory):

    value: float
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import LowerThen as LTTransformation

        return LTTransformation(self.in_feature.name, self.value)


@dataclass
class LowerThenOrEqualFactory(TransformationFactory):

    value: float
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import LowerThenOrEqual as LTETransformation

        return LTETransformation(self.in_feature.name, self.value)


# @dataclass
# class Split(TransformationFactory):

#     pattern: str
#     from_feature: FeatureFactory
#     max_splits: int | None

#     @property
#     def method(self) -> Callable[[DataFrame], Series]:
#         async def met(df: DataFrame) -> Series:
#             return df[self.from_feature.name].str.split(pat=self.pattern, n=self.max_splits)

#         return met

#     def index(self, index: int) -> 'ArrayIndex':
#         return ArrayIndex(index, self)


# class ArrayIndex(DillTransformationFactory):

#     index: int
#     from_feature: FeatureFactory

#     def __init__(self, index: int, feature: FeatureFactory) -> None:
#         self.using_features = [feature]
#         self.feature = Bool()
#         self.index = index
#         self.from_feature = feature

#     @property
#     def method(self) -> Callable[[DataFrame], Series]:
#         async def met(df: DataFrame) -> Series:
#             return df[self.from_feature.name].str[self.index]

#         return met


@dataclass
class DateComponentFactory(TransformationFactory):

    component: str
    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import DateComponent as DCTransformation

        return DCTransformation(self.feature.name, self.component)


@dataclass
class DifferanceBetweenFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Subtraction

        return Subtraction(self.first_feature.name, self.second_feature.name)


@dataclass
class AdditionBetweenFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Addition

        return Addition(self.first_feature.name, self.second_feature.name)


@dataclass
class TimeDifferanceFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import TimeDifference as TDTransformation

        return TDTransformation(self.first_feature.name, self.second_feature.name)


@dataclass
class LogTransformFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import LogarithmOnePluss

        return LogarithmOnePluss(self.feature.name)


@dataclass
class ReplaceFactory(TransformationFactory):

    values: dict[str, str]
    source_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.source_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import ReplaceStrings

        return ReplaceStrings(self.source_feature.name, self.values)


@dataclass
class ToNumericalFactory(TransformationFactory):

    from_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.from_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import ToNumerical as ToNumericalTransformation

        return ToNumericalTransformation(self.from_feature.name)


@dataclass
class StandardScalingFactory(TransformationFactory):

    feature: FeatureFactory

    limit: int | None = field(default=None)
    timespan: timedelta | None = field(default=None)

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.enricher import StatisticEricher

        if self.feature.transformation:
            raise ValueError('Standard scaling is not supported for derived features yet')

        if len(source_views) != 1:
            raise ValueError('Expected one source')

        feature_view = source_views[0]

        if not isinstance(feature_view.batch_data_source, StatisticEricher):
            raise ValueError('The data source needs to conform to StatisticEricher')

        timespan: TimespanSelector | None = None

        feature_name = self.feature.name

        if self.timespan:
            if not feature_view.event_timestamp:
                raise InvalidStandardScalerArtefact(
                    'Unable to find a event_timestamp, this is needed'
                    ' when using `timespan` for artefact generation.\n',
                    'Make sure the event_timestamp is above the'
                    ' transformation in the feature view decleration',
                )
            timespan = TimespanSelector(self.timespan, time_column=feature_view.event_timestamp.name)

        std_enricher = feature_view.batch_data_source.std(
            columns={feature_name}, time=timespan, limit=self.limit
        )
        mean_enricher = feature_view.batch_data_source.mean(
            columns={feature_name}, time=timespan, limit=self.limit
        )

        std, mean = await asyncio.gather(std_enricher.as_df(), mean_enricher.as_df())

        if std.isna().any() or (std == 0).any():
            raise InvalidStandardScalerArtefact(
                f'The standard deviation for {feature_name} is 0.'
                'Therefore convaying no meaningful information.\n'
                'This could be because the used dataset has no values,'
                'so maybe consider changing `limit`,`timspan` or change the datasource'
            )

        return StandardScalingTransformation(mean[feature_name], std[feature_name], feature_name)


@dataclass
class IsInFactory(TransformationFactory):

    feature: FeatureFactory
    values: list

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import IsIn as IsInTransformation

        return IsInTransformation(self.values, self.feature.name)


@dataclass
class AndFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import And as AndTransformation

        return AndTransformation(self.first_feature.name, self.second_feature.name)


@dataclass
class OrFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Or as OrTransformation

        return OrTransformation(self.first_feature.name, self.second_feature.name)


@dataclass
class InverseFactory(TransformationFactory):

    from_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.from_feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Inverse as InverseTransformation

        return InverseTransformation(self.from_feature.name)


class FillNaStrategy:
    async def compile(self, feature: FeatureFactory, source_views: list[CompiledFeatureView]) -> Any:
        pass

    @staticmethod
    def mean(limit: int | None = None) -> 'FillNaStrategy':
        return MeanFillNaStrategy(limit)


@dataclass
class ConstantFillNaStrategy(FillNaStrategy):
    value: Any

    async def compile(self, feature: FeatureFactory, source_views: list[CompiledFeatureView]) -> Any:
        return self.value


@dataclass
class MeanFillNaStrategy(FillNaStrategy):

    limit: int | None = field(default=None)

    async def compile(self, feature: FeatureFactory, source_views: list[CompiledFeatureView]) -> Any:
        if len(source_views) != 1:
            raise ValueError('Need exactly one source in order to compute mean fill value')

        source = source_views[0]
        if not isinstance(source.batch_data_source, StatisticEricher):
            raise ValueError('The data source needs to be a StatisticEnricher')

        mean = await source.batch_data_source.mean(columns={feature.name}, limit=self.limit).as_df()
        return mean[feature.name]


@dataclass
class FillMissingFactory(TransformationFactory):

    feature: FeatureFactory
    strategy: FillNaStrategy

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import FillNaValues

        fill_value = await self.strategy.compile(self.feature, source_views)
        return FillNaValues(key=self.feature.name, value=fill_value, dtype=self.feature.dtype)


@dataclass
class FloorFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Floor

        return Floor(self.feature.name)


@dataclass
class CeilFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Ceil

        return Ceil(self.feature.name)


@dataclass
class RoundFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Round

        return Round(self.feature.name)


@dataclass
class AbsoluteFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Absolute

        return Absolute(self.feature.name)


@dataclass
class DillTransformationFactory(TransformationFactory):

    dtype: FeatureFactory
    method: Callable[[DataFrame], Series]
    _using_features: list[FeatureFactory]

    @property
    def using_features(self) -> list[FeatureFactory]:
        return self._using_features

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        import dill

        from aligned.schemas.transformation import DillTransformation

        return DillTransformation(method=dill.dumps(self.method, recurse=True), dtype=self.dtype.dtype)


class AggregatableTransformation:

    group_by: list[FeatureFactory] | None = field(default=None)

    def copy(self) -> 'AggregatableTransformation':
        pass


@dataclass
class MeanTransfomrationFactory(TransformationFactory, AggregatableTransformation):

    feature: FeatureFactory
    over: timedelta | None = field(default=None)
    group_by: list[FeatureFactory] | None = field(default=None)

    @property
    def using_features(self) -> list[FeatureFactory]:
        if self.group_by:
            return [self.feature] + self.group_by
        else:
            return [self.feature]

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        from aligned.schemas.transformation import Mean

        if len(source_views) != 1:
            raise ValueError('Unable to compute mean for CombinedView')

        return Mean(
            key=self.feature.name, group_keys=[feat.name for feat in self.group_by] if self.group_by else None
        )

    def copy(self) -> 'MeanTransfomrationFactory':
        return MeanTransfomrationFactory(self.feature, self.over, self.group_by)
