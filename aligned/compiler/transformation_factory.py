import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

import pandas as pd
import polars as pl

from aligned.compiler.feature_factory import FeatureFactory, Transformation, TransformationFactory
from aligned.schemas.transformation import LiteralValue, TextVectoriserModel

logger = logging.getLogger(__name__)


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

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Equals

        if isinstance(self.right, FeatureFactory):
            raise NotImplementedError()
        else:
            return Equals(self.left.name, LiteralValue.from_value(self.right))


@dataclass
class NotNullFactory(TransformationFactory):

    value: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.value]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import NotNull

        return NotNull(self.value.name)


@dataclass
class RatioFactory(TransformationFactory):

    numerator: FeatureFactory
    denumerator: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.numerator, self.denumerator]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Ratio

        return Ratio(self.numerator.name, self.denumerator.name)


@dataclass
class OrdinalFactory(TransformationFactory):

    orders: list[str]
    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Ordinal

        return Ordinal(self.feature.name, self.orders)


@dataclass
class ContainsFactory(TransformationFactory):

    text: str
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Contains as ContainsTransformation

        return ContainsTransformation(self.in_feature.name, self.text)


@dataclass
class NotEqualsFactory(TransformationFactory):

    value: Any
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import NotEquals as NotEqualsTransformation

        return NotEqualsTransformation(self.in_feature.name, LiteralValue.from_value(self.value))


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

    def compile(self) -> Transformation:
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

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import GreaterThenOrEqual as GTETransformation

        return GTETransformation(self.in_feature.name, self.value)


@dataclass
class LowerThenFactory(TransformationFactory):

    value: float
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import LowerThen as LTTransformation

        return LTTransformation(self.in_feature.name, self.value)


@dataclass
class LowerThenOrEqualFactory(TransformationFactory):

    value: float
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import LowerThenOrEqual as LTETransformation

        return LTETransformation(self.in_feature.name, self.value)


# @dataclass
# class Split(TransformationFactory):

#     pattern: str
#     from_feature: FeatureFactory
#     max_splits: int | None

#     @property
#     def method(self) -> Callable[[pd.DataFrame], pd.Series]:
#         async def met(df: pd.DataFrame) -> pd.Series:
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
#     def method(self) -> Callable[[pd.DataFrame], pd.Series]:
#         async def met(df: pd.DataFrame) -> pd.Series:
#             return df[self.from_feature.name].str[self.index]

#         return met


@dataclass
class DateComponentFactory(TransformationFactory):

    component: str
    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import DateComponent as DCTransformation

        return DCTransformation(self.feature.name, self.component)


@dataclass
class DifferanceBetweenFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Subtraction

        return Subtraction(self.first_feature.name, self.second_feature.name)


@dataclass
class AdditionBetweenFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Addition

        return Addition(self.first_feature.name, self.second_feature.name)


@dataclass
class PowerFactory(TransformationFactory):

    first: FeatureFactory
    second: FeatureFactory | Any

    @property
    def using_features(self) -> list[FeatureFactory]:
        if isinstance(self.second, FeatureFactory):
            return [self.first, self.second]
        return [self.first]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Power, PowerFeature

        if isinstance(self.second, FeatureFactory):
            return PowerFeature(self.first.name, self.second.name)
        else:
            value = LiteralValue.from_value(self.second)
            return Power(self.first.name, value)


@dataclass
class TimeDifferanceFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import TimeDifference as TDTransformation

        return TDTransformation(self.first_feature.name, self.second_feature.name)


@dataclass
class LogTransformFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import LogarithmOnePluss

        return LogarithmOnePluss(self.feature.name)


@dataclass
class ReplaceFactory(TransformationFactory):

    values: dict[str, str]
    source_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.source_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import ReplaceStrings

        return ReplaceStrings(self.source_feature.name, self.values)


@dataclass
class ToNumericalFactory(TransformationFactory):

    from_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.from_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import ToNumerical as ToNumericalTransformation

        return ToNumericalTransformation(self.from_feature.name)


@dataclass
class IsInFactory(TransformationFactory):

    feature: FeatureFactory
    values: list

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import IsIn as IsInTransformation

        return IsInTransformation(self.values, self.feature.name)


@dataclass
class AndFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import And as AndTransformation

        return AndTransformation(self.first_feature.name, self.second_feature.name)


@dataclass
class OrFactory(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Or as OrTransformation

        return OrTransformation(self.first_feature.name, self.second_feature.name)


@dataclass
class InverseFactory(TransformationFactory):

    from_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.from_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Inverse as InverseTransformation

        return InverseTransformation(self.from_feature.name)


class FillNaStrategy:
    def compile(self) -> Any:
        pass

    @staticmethod
    def mean(limit: int | None = None) -> 'FillNaStrategy':
        return MeanFillNaStrategy(limit)


@dataclass
class ConstantFillNaStrategy(FillNaStrategy):
    value: Any

    def compile(self) -> Any:
        return self.value


@dataclass
class MeanFillNaStrategy(FillNaStrategy):

    limit: int | None = field(default=None)

    def compile(self) -> Any:
        logging.info('The Mean Fill strategy will be deprecated.')
        return 0


@dataclass
class FillMissingFactory(TransformationFactory):

    feature: FeatureFactory
    strategy: FillNaStrategy

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import FillNaValues

        fill_value = self.strategy.compile()

        return FillNaValues(
            key=self.feature.name, value=LiteralValue.from_value(fill_value), dtype=self.feature.dtype
        )


@dataclass
class FloorFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Floor

        return Floor(self.feature.name)


@dataclass
class CeilFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Ceil

        return Ceil(self.feature.name)


@dataclass
class RoundFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Round

        return Round(self.feature.name)


@dataclass
class AbsoluteFactory(TransformationFactory):

    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Absolute

        return Absolute(self.feature.name)


@dataclass
class PandasTransformationFactory(TransformationFactory):

    dtype: FeatureFactory
    method: Callable[[pd.DataFrame], pd.Series]
    _using_features: list[FeatureFactory]

    @property
    def using_features(self) -> list[FeatureFactory]:
        return self._using_features

    def compile(self) -> Transformation:
        import types

        import dill

        from aligned.schemas.transformation import PandasFunctionTransformation, PandasLambdaTransformation

        if isinstance(self.method, types.LambdaType) and self.method.__name__ == '<lambda>':
            return PandasLambdaTransformation(
                method=dill.dumps(self.method),
                code=dill.source.getsource(self.method),
                dtype=self.dtype.dtype,
            )
        else:
            return PandasFunctionTransformation(
                code=dill.source.getsource(self.method),
                function_name=dill.source.getname(self.method),
                dtype=self.dtype.dtype,
            )


@dataclass
class PolarsTransformationFactory(TransformationFactory):

    dtype: FeatureFactory
    method: pl.Expr | Callable[[pl.LazyFrame, pl.Expr], pl.LazyFrame]
    _using_features: list[FeatureFactory]

    @property
    def using_features(self) -> list[FeatureFactory]:
        return self._using_features

    def compile(self) -> Transformation:
        import types

        import dill

        from aligned.schemas.transformation import PolarsFunctionTransformation, PolarsLambdaTransformation

        code = dill.source.getsource(self.method)

        if isinstance(self.method, pl.Expr):
            code = str(self.method)
            self.method = lambda df, alias: df.with_column(self.method.alias(alias))

        if isinstance(self.method, types.LambdaType) and self.method.__name__ == '<lambda>':
            return PolarsLambdaTransformation(
                method=dill.dumps(self.method), code=code, dtype=self.dtype.dtype
            )
        else:
            return PolarsFunctionTransformation(
                code=code,
                function_name=dill.source.getname(self.method),
                dtype=self.dtype.dtype,
            )


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

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Mean

        return Mean(
            key=self.feature.name, group_keys=[feat.name for feat in self.group_by] if self.group_by else None
        )

    def copy(self) -> 'MeanTransfomrationFactory':
        return MeanTransfomrationFactory(self.feature, self.over, self.group_by)


@dataclass
class WordVectoriserFactory(TransformationFactory):

    feature: FeatureFactory
    model: TextVectoriserModel

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import WordVectoriser

        return WordVectoriser(self.feature.name, self.model)


@dataclass
class LoadImageFactory(TransformationFactory):

    url_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.url_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import LoadImageUrl

        return LoadImageUrl(self.url_feature.name)


@dataclass
class GrayscaleImageFactory(TransformationFactory):

    image_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.image_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import GrayscaleImage

        return GrayscaleImage(self.image_feature.name)


@dataclass
class AppendStrings(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory | LiteralValue

    @property
    def using_features(self) -> list[FeatureFactory]:
        if isinstance(self.second_feature, LiteralValue):
            return [self.first_feature]
        else:
            return [self.first_feature, self.second_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import AppendConstString, AppendStrings

        if isinstance(self.second_feature, LiteralValue):
            return AppendConstString(self.first_feature.name, self.second_feature.value)
        else:
            return AppendStrings(self.first_feature.name, self.second_feature.name)
