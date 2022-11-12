from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from pandas import DataFrame, Series

from aligned.compiler.constraint_factory import ConstraintFactory, LiteralFactory
from aligned.exceptions import NotSupportedYet
from aligned.schemas.constraints import (
    Constraint,
    InDomain,
    LowerBound,
    LowerBoundInclusive,
    UpperBound,
    UpperBoundInclusive,
)
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import EventTimestamp as EventTimestampFeature
from aligned.schemas.feature import Feature, FeatureReferance, FeatureType
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.schemas.transformation import Transformation

if TYPE_CHECKING:
    from aligned.compiler.transformation_factory import FillNaStrategy


class TransformationFactory:
    """
    A class that can compute the transformation logic.

    For most classes will there be no need for a factory,
    However for more advanced transformations will this be needed.

    e.g:

    StandaredScalerFactory(
        time_window=timedelta(days=30)
    )

    The batch data source will be provided when the compile method is run.
    leading to fetching a small sample, and compute the metrics needed in order to generate a

    StandardScalerTransformation config. E.g:
    StandardScalerTransformation(
        mean=10,
        std=0.3
    )
    """

    async def compile(self, source_views: list[CompiledFeatureView]) -> Transformation:
        pass

    @property
    def using_features(self) -> list[FeatureFactory]:
        pass


T = TypeVar('T')


class FeatureFactory:
    """
    Represents the information needed to generate a feature definition

    This may contain lazely loaded information, such as then name.
    Which will be added when the feature view is compiled, as we can get the attribute name at runtime.

    The feature can still have no name, but this means that it is an unstored feature.
    It will threfore need a transformation field.

    The feature_dependencies is the features graph for the given feature.

    aka
                            x <- standard scaler <- age: Float
    x_and_y_is_equal <-
                            y: Float
    """

    _name: str | None = None
    _feature_view: str | None = None
    _description: str | None = None

    transformation: TransformationFactory | None = None
    constraints: set[ConstraintFactory] | None = None

    @property
    def dtype(self) -> FeatureType:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        if not self._name:
            raise ValueError('Have not been given a name yet')
        return self._name

    @property
    def depending_on_names(self) -> list[str]:
        if not self.transformation:
            return []
        return [feat._name for feat in self.transformation.using_features if feat._name]

    def feature_referance(self) -> FeatureReferance:
        return FeatureReferance(self.name, self._feature_view, self.dtype, self.transformation is not None)

    async def feature(self) -> Feature:
        return Feature(
            name=self.name,
            dtype=self.dtype,
            description=self._description,
            tags=None,
            constraints=self.constraints,
        )

    def compile_graph_only(self) -> DerivedFeature:
        from aligned.schemas.transformation import CopyTransformation

        if not self.transformation:
            raise ValueError('Trying to create a derived feature with no transformation')

        some_feature = self.transformation.using_features[0]
        return DerivedFeature(
            name=self.name,
            dtype=self.dtype,
            depending_on=[feat.feature_referance() for feat in self.transformation.using_features],
            transformation=CopyTransformation(some_feature.name, dtype=some_feature.dtype),
            depth=self.depth(),
            description=self._description,
            tags=None,
            constraints=None,
        )

    async def compile(self, source_views: list[CompiledFeatureView]) -> DerivedFeature:

        if not self.transformation:
            raise ValueError('Trying to create a derived feature with no transformation')

        return DerivedFeature(
            name=self.name,
            dtype=self.dtype,
            depending_on=[feat.feature_referance() for feat in self.transformation.using_features],
            transformation=await self.transformation.compile(source_views),
            depth=self.depth(),
            description=self._description,
            tags=None,
            constraints=None,
        )

    def depth(self) -> int:
        value = 0
        if not self.transformation:
            return value
        for feature in self.transformation.using_features:
            value = max(feature.depth(), value)
        return value + 1

    def description(self: T, description: str) -> T:
        self._description = description  # type: ignore [attr-defined]
        return self

    def feature_dependencies(self) -> list[FeatureFactory]:
        values = []

        if not self.transformation:
            return []

        def add_values(feature: FeatureFactory) -> None:
            values.append(feature)
            if not feature.transformation:
                return
            for sub_feature in feature.transformation.using_features:
                add_values(sub_feature)

        for sub_feature in self.transformation.using_features:
            add_values(sub_feature)

        return values

    def copy_type(self: T) -> T:
        raise NotImplementedError()

    def fill_na(self: T, value: FillNaStrategy | Any) -> T:

        from aligned.compiler.transformation_factory import (
            ConstantFillNaStrategy,
            FillMissingFactory,
            FillNaStrategy,
        )

        instance: FeatureFactory = self.copy_type()  # type: ignore [attr-defined]
        if isinstance(value, FillNaStrategy):
            instance.transformation = FillMissingFactory(self, value)
        else:
            instance.transformation = FillMissingFactory(self, ConstantFillNaStrategy(value))
        return instance  # type: ignore [return-value]

    def transformed(
        self,
        transformation: Callable[[DataFrame], Series],
        using_features: list[FeatureFactory] | None = None,
        as_dtype: T | None = None,
    ) -> T:
        import asyncio

        from aligned.compiler.transformation_factory import DillTransformationFactory

        dtype: FeatureFactory = as_dtype or self.copy_type()  # type: ignore [assignment]

        if asyncio.iscoroutinefunction(transformation):
            dtype.transformation = DillTransformationFactory(dtype, transformation, using_features or [self])
        else:

            async def sub_tran(df: DataFrame) -> Series:
                return transformation(df)

            dtype.transformation = DillTransformationFactory(dtype, sub_tran, using_features or [self])
        return dtype  # type: ignore [return-value]

    def is_required(self: T) -> T:
        from aligned.schemas.constraints import Required

        self._add_constraint(Required())  # type: ignore[attr-defined]
        return self

    def _add_constraint(self, constraint: ConstraintFactory | Constraint) -> None:
        # The constraint should be a lazy evaluated constraint
        # Aka, a factory, as with the features.
        # Therefore making it possible to add distribution checks
        if not self.constraints:
            self.constraints = set()
        if isinstance(constraint, Constraint):
            self.constraints.add(constraint)
        else:
            self.constraints.add(LiteralFactory(constraint))


class EquatableFeature(FeatureFactory):

    # Comparable operators
    def __eq__(self, right: FeatureFactory | Any) -> Bool:  # type: ignore[override]
        from aligned.compiler.transformation_factory import EqualsFactory

        instance = Bool()
        instance.transformation = EqualsFactory(self, right)
        return instance

    def equals(self, right: object) -> Bool:
        return self == right

    def __ne__(self, right: FeatureFactory | Any) -> Bool:  # type: ignore[override]
        from aligned.compiler.transformation_factory import NotEqualsFactory

        instance = Bool()
        instance.transformation = NotEqualsFactory(right, self)
        return instance

    def not_equals(self, right: object) -> Bool:
        return self != right

    def is_in(self, values: list[Any]) -> Bool:
        from aligned.compiler.transformation_factory import IsInFactory

        instance = Bool()
        instance.transformation = IsInFactory(self, values)
        return instance


class ComparableFeature(EquatableFeature):
    def __lt__(self, right: object) -> Bool:
        from aligned.compiler.transformation_factory import LowerThenFactory

        instance = Bool()
        instance.transformation = LowerThenFactory(right, self)
        return instance

    def __le__(self, right: float) -> Bool:
        from aligned.compiler.transformation_factory import LowerThenOrEqualFactory

        instance = Bool()
        instance.transformation = LowerThenOrEqualFactory(right, self)
        return instance

    def __gt__(self, right: object) -> Bool:
        from aligned.compiler.transformation_factory import GreaterThenFactory

        instance = Bool()
        instance.transformation = GreaterThenFactory(right, self)
        return instance

    def __ge__(self, right: object) -> Bool:
        from aligned.compiler.transformation_factory import GreaterThenOrEqualFactory

        instance = Bool()
        instance.transformation = GreaterThenOrEqualFactory(right, self)
        return instance

    def lower_bound(self: T, value: float, is_inclusive: bool | None = None) -> T:

        if is_inclusive:
            self._add_constraint(LowerBoundInclusive(value))  # type: ignore[attr-defined]
        else:
            self._add_constraint(LowerBound(value))  # type: ignore[attr-defined]
        return self

    def upper_bound(self: T, value: float, is_inclusive: bool | None = None) -> T:

        if is_inclusive:
            self._add_constraint(UpperBoundInclusive(value))  # type: ignore[attr-defined]
        else:
            self._add_constraint(UpperBound(value))  # type: ignore[attr-defined]
        return self


class ArithmeticFeature(ComparableFeature):
    def __sub__(self, other: FeatureFactory) -> Float:
        from aligned.compiler.transformation_factory import DifferanceBetweenFactory, TimeDifferanceFactory

        feature = Float()
        if self.dtype == FeatureType('').datetime:
            feature.transformation = TimeDifferanceFactory(self, other)
        else:
            feature.transformation = DifferanceBetweenFactory(self, other)
        return feature

    def __add__(self, other: FeatureFactory) -> Float:
        from aligned.compiler.transformation_factory import AdditionBetweenFactory

        feature = Float()
        feature.transformation = AdditionBetweenFactory(self, other)
        return feature

    def __truediv__(self, other: FeatureFactory) -> Float:
        from aligned.compiler.transformation_factory import RatioFactory

        feature = Float()
        feature.transformation = RatioFactory(self, other)
        return feature

    def __floordiv__(self, other: FeatureFactory) -> Float:
        from aligned.compiler.transformation_factory import RatioFactory

        feature = Float()
        feature.transformation = RatioFactory(self, other)
        return feature

    def __abs__(self) -> Float:
        from aligned.compiler.transformation_factory import AbsoluteFactory

        feature = Float()
        feature.transformation = AbsoluteFactory(self)
        return feature

    def standard_scaled(self, timespan: timedelta | None = None, limit: int | None = None) -> Float:
        from aligned.compiler.transformation_factory import StandardScalingFactory

        feature = Float()
        feature.transformation = StandardScalingFactory(feature=self, limit=limit, timespan=timespan)
        return feature

    def log1p(self) -> Float:
        from aligned.compiler.transformation_factory import LogTransformFactory

        feature = Float()
        feature.transformation = LogTransformFactory(self)
        return feature

    def mean(self: T, over: timedelta | None = None) -> NumericalAggregation[T]:
        from aligned.compiler.transformation_factory import MeanTransfomrationFactory

        if over:
            raise NotSupportedYet('Computing mean with a time window is not supported yet')
        feature = NumericalAggregation(self)
        feature.transformation = MeanTransfomrationFactory(self)
        return feature


class DecimalOperations(FeatureFactory):
    def __round__(self) -> Int64:
        from aligned.compiler.transformation_factory import RoundFactory

        feature = Int64()
        feature.transformation = RoundFactory(self)
        return feature

    def __ceil__(self) -> Int64:
        from aligned.compiler.transformation_factory import CeilFactory

        feature = Int64()
        feature.transformation = CeilFactory(self)
        return feature

    def __floor__(self) -> Int64:
        from aligned.compiler.transformation_factory import FloorFactory

        feature = Int64()
        feature.transformation = FloorFactory(self)
        return feature


class TruncatableFeature(FeatureFactory):
    def __trunc__(self: T) -> T:
        raise NotImplementedError()


class NumberConvertableFeature(FeatureFactory):
    def as_float(self) -> Float:
        from aligned.compiler.transformation_factory import ToNumericalFactory

        feature = Float()
        feature.transformation = ToNumericalFactory(self)
        return feature

    def __int__(self) -> Int64:
        raise NotImplementedError()

    def __float__(self) -> Float:
        raise NotImplementedError()


class InvertableFeature(FeatureFactory):
    def __invert__(self) -> Bool:
        from aligned.compiler.transformation_factory import InverseFactory

        feature = Bool()
        feature.transformation = InverseFactory(self)
        return feature


class LogicalOperatableFeature(InvertableFeature):
    def __and__(self, other: Bool) -> Bool:
        from aligned.compiler.transformation_factory import AndFactory

        feature = Bool()
        feature.transformation = AndFactory(self, other)
        return feature

    def logical_and(self, other: Bool) -> Bool:
        return self & other

    def __or__(self, other: Bool) -> Bool:
        from aligned.compiler.transformation_factory import OrFactory

        feature = Bool()
        feature.transformation = OrFactory(self, other)
        return feature

    def logical_or(self, other: Bool) -> Bool:
        return self | other


class CategoricalEncodableFeature(EquatableFeature):
    def one_hot_encode(self, labels: list[str]) -> list[Bool]:
        return [self == label for label in labels]

    def ordinal_categories(self, orders: list[str]) -> Int32:
        from aligned.compiler.transformation_factory import OrdinalFactory

        feature = Int32()
        feature.transformation = OrdinalFactory(orders, self)
        return feature

    def accepted_values(self: T, values: list[str]) -> T:
        self._add_constraint(InDomain(values))  # type: ignore[attr-defined]
        return self


class DateFeature(FeatureFactory):
    def date_component(self, component: str) -> Int32:
        from aligned.compiler.transformation_factory import DateComponentFactory

        feature = Int32()
        feature.transformation = DateComponentFactory(component, self)
        return feature


class Bool(EquatableFeature, LogicalOperatableFeature):
    @property
    def dtype(self) -> FeatureType:
        return FeatureType('').bool

    def copy_type(self) -> Bool:
        return Bool()


class Float(ArithmeticFeature, DecimalOperations):
    def copy_type(self) -> Float:
        return Float()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType('').float


NumericType = TypeVar('NumericType', bound=ArithmeticFeature)


class NumericalAggregation(Generic[NumericType], ArithmeticFeature, DecimalOperations):
    def __init__(self, dtype: NumericType):
        self._dtype = dtype

    _dtype: NumericType

    def copy_type(self: NumericalAggregation) -> NumericalAggregation:
        return NumericalAggregation()

    @property
    def dtype(self) -> FeatureType:
        return self._dtype.dtype

    def grouped_by(self, keys: FeatureFactory | list[FeatureFactory]) -> NumericType:
        from aligned.compiler.transformation_factory import AggregatableTransformation

        if not isinstance(self.transformation, AggregatableTransformation):
            raise ValueError(
                f'Can only group by on aggregatable transformations. This is a {self.transformation}'
            )

        feature = self._dtype.copy_type()
        feature.transformation = self.transformation.copy()
        feature.transformation.group_by = keys if isinstance(keys, list) else [keys]
        return feature


class Int32(ArithmeticFeature):
    def copy_type(self) -> Int32:
        return Int32()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType('').int32


class Int64(ArithmeticFeature):
    def copy_type(self) -> Int64:
        return Int64()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType('').int64


class UUID(FeatureFactory):
    def copy_type(self) -> UUID:
        return UUID()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType('').uuid


class String(CategoricalEncodableFeature, NumberConvertableFeature):
    def copy_type(self) -> String:
        return String()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType('').string

    def split(self, pattern: str, max_splits: int | None = None) -> String:
        raise NotImplementedError()

    def replace(self, values: dict[str, str]) -> String:
        from aligned.compiler.transformation_factory import ReplaceFactory

        feature = String()
        feature.transformation = ReplaceFactory(values, self)
        return feature

    def contains(self, value: str) -> Bool:
        from aligned.compiler.transformation_factory import ContainsFactory

        feature = Bool()
        feature.transformation = ContainsFactory(value, self)
        return feature


class Entity(FeatureFactory):

    _dtype: FeatureFactory

    @property
    def dtype(self) -> FeatureType:
        return self._dtype.dtype

    def __init__(self, dtype: FeatureFactory):
        self._dtype = dtype


class Timestamp(DateFeature, ArithmeticFeature):
    @property
    def dtype(self) -> FeatureType:
        return FeatureType('').datetime


class EventTimestamp(DateFeature, ArithmeticFeature):

    ttl: timedelta | None

    @property
    def dtype(self) -> FeatureType:
        return FeatureType('').datetime

    def __init__(self, ttl: timedelta | None = None):
        self.ttl = ttl

    def event_timestamp(self) -> EventTimestampFeature:
        return EventTimestampFeature(
            name=self.name, ttl=self.ttl.total_seconds() if self.ttl else None, description=self._description
        )
