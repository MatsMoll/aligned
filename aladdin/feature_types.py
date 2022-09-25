from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Literal, TypeVar
from uuid import uuid4

from pandas import DataFrame, Series

from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import Constraint
from aladdin.feature import EventTimestamp as EventTimestampFeature
from aladdin.feature import Feature, FeatureReferance, FeatureType
from aladdin.feature_view.feature_view_metadata import FeatureViewMetadata
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.transformation import StandardScalingTransformation, Transformation


class FeatureReferancable:

    _name: str | None = None
    _generated_name: str | None = None
    _feature_view: str | None = None

    @property
    def is_derived(self) -> bool:
        return isinstance(self, TransformationFactory)

    @property
    def name(self) -> str:
        if not self._name and not self._generated_name:
            self._generated_name = str(uuid4())
        return self._name or self._generated_name  # type: ignore

    @property
    def feature_view(self) -> str:
        if not self._feature_view:
            raise ValueError('Feature view is not set')
        return self._feature_view

    @abstractproperty
    def _dtype(self) -> FeatureType:
        pass

    def copy_dtype(self) -> 'FeatureReferancable':
        values: dict[str, FeatureReferancable] = {
            'string': String(),
            'int32': Int32(),
            'int64': Int64(),
            'float': Float(),
            'double': Double(),
            'bool': Bool(),
            'array': Array(),
        }
        return values[self._dtype.name]

    def feature_referance(self) -> FeatureReferance:
        if not self.name or not self.feature_view:
            raise ValueError('name and feature_view must be set')
        return FeatureReferance(
            name=self.name,
            feature_view=self.feature_view,
            dtype=self._dtype,
            is_derivied=self.is_derived,
        )


class Transformable(FeatureReferancable):
    def transformed(
        self,
        transformation: Callable[[DataFrame], Series],
        using_features: list[FeatureReferancable] | None = None,
        as_dtype: FeatureReferancable | None = None,
    ) -> 'TransformationFactory':
        """
        Defines a custom python transformation

        Args:
            transformation (Callable[[DataFrame], Series]): The transformation to perform
            using_features (list[FeatureReferancable] | None, optional):
                The features that this feature depends on. Defaults to None.
            as_dtype (FeatureReferancable | None, optional): The data type of the feature. Defaults to None.

        Returns:
            TransformationFactory: The transformed feature
        """
        import asyncio

        dtype = as_dtype or self.copy_dtype()

        if asyncio.iscoroutinefunction(transformation):
            return CustomTransformation(using_features or [self], dtype, transformation)

        async def sub_tran(df: DataFrame) -> Series:
            return transformation(df)

        return CustomTransformation(using_features or [self], dtype, sub_tran)

    # Comparable operators
    def __eq__(self, __o: object) -> 'Equals':  # type: ignore
        return Equals(__o, self)

    def __ne__(self, __o: object) -> 'NotEquals':  # type: ignore
        return NotEquals(__o, self)

    def __lt__(self, value: float) -> 'LowerThen':
        return LowerThen(value, self)

    def __le__(self, value: float) -> 'LowerThenOrEqual':
        return LowerThenOrEqual(value, self)

    def __gt__(self, value: object) -> 'GreaterThen':
        return GreaterThen(value, self)

    def __ge__(self, value: object) -> 'GreaterThenOrEqual':
        return GreaterThenOrEqual(value, self)

    # Arithmetic Operators
    def __sub__(self, other: FeatureReferancable) -> 'Transformable':
        if self._dtype == FeatureType('').datetime:
            return TimeDifferance(self, other)
        return DifferanceBetween(self, other)

    def __add__(self, other: FeatureReferancable) -> 'AdditionBetween':
        return AdditionBetween(self, other)

    def __truediv__(self, other: FeatureReferancable) -> 'Ratio':
        return Ratio(self, other)

    def __floordiv__(self, other: FeatureReferancable) -> 'Ratio':
        raise NotImplementedError()

    # Bool operators
    def __and__(self, other: FeatureReferancable) -> 'And':
        return And(self, other)

    def __or__(self, other: FeatureReferancable) -> 'Or':
        return Or(self, other)

    def is_in(self, values: list) -> 'IsIn':
        return IsIn(self, values)

    # Single value operators
    def __invert__(self) -> 'Inverse':
        return Inverse(self)

    def __abs__(self) -> 'FeatureReferancable':
        raise NotImplementedError()

    def __int__(self) -> 'FeatureReferancable':
        raise NotImplementedError()

    def __float__(self) -> 'FeatureReferancable':
        raise NotImplementedError()

    def __complex__(self) -> 'FeatureReferancable':
        raise NotImplementedError()

    def __round__(self) -> 'FeatureReferancable':
        raise NotImplementedError()

    def __trunc__(self) -> 'FeatureReferancable':
        raise NotImplementedError()

    def __ceil__(self) -> 'FeatureReferancable':
        raise NotImplementedError()

    def __floor__(self) -> 'FeatureReferancable':
        raise NotImplementedError()

    def fill_missing(self, strategy: Literal['mean', 'median']) -> 'Transformable':
        return FillMissing(self, strategy)


FeatureTypeVar = TypeVar('FeatureTypeVar')


class FeatureFactory(ABC, Transformable):

    _labels: dict[str, str] | None = None
    _description: str | None = None

    constraints: set[Constraint] = set()

    @abstractproperty
    def _dtype(self) -> FeatureType:
        pass

    def feature(self, name: str) -> Feature:
        self._name = name
        return Feature(
            name,
            dtype=self._dtype,
            description=self._description,
            tags=self._labels,
            constraints=self.constraints,
        )

    def labels(self: FeatureTypeVar, labels: dict[str, str]) -> FeatureTypeVar:
        if isinstance(self, FeatureFactory):
            self._labels = labels
        return self

    def description(self: FeatureTypeVar, description: str) -> FeatureTypeVar:
        if isinstance(self, FeatureFactory):
            self._description = description
        return self


@dataclass
class CustomTransformationV2(Transformation):

    method: bytes
    dtype: FeatureType
    name: str = 'custom_transformation'

    @staticmethod
    def with_method(method: Callable[[DataFrame], Series], dtype: FeatureType) -> 'CustomTransformationV2':
        import dill

        return CustomTransformationV2(method=dill.dumps(method, recurse=True), dtype=dtype)

    async def transform(self, df: DataFrame) -> Series:
        import dill

        loaded = dill.loads(self.method)
        return await loaded(df)


class TransformationFactory(ABC, Transformable):

    using_features: list[FeatureReferancable]
    feature: FeatureReferancable

    @property
    def _dtype(self) -> FeatureType:
        return self.feature._dtype

    @abstractmethod
    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        pass

    @property
    def using_feature_names(self) -> list[str]:
        return [feature.name for feature in self.using_features]

    def labels(self: FeatureTypeVar, labels: dict[str, str]) -> FeatureTypeVar:
        if isinstance(self, TransformationFactory) and isinstance(self.feature, FeatureFactory):
            self.feature._labels = labels
        return self

    def description(self: FeatureTypeVar, description: str) -> FeatureTypeVar:
        if isinstance(self, TransformationFactory) and isinstance(self.feature, FeatureFactory):
            self.feature._description = description
        return self

    def derived_feature(
        self, name: str, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]
    ) -> DerivedFeature:

        feature: Feature
        if isinstance(self.feature, FeatureFactory):
            feature = self.feature.feature(name)
        else:
            feature = Feature(
                self.feature.name, self.feature._dtype, description=None, tags=None, constraints=None
            )

        return DerivedFeature(
            name=feature.name,
            dtype=feature.dtype,
            depending_on={feat.feature_referance() for feat in self.using_features},
            transformation=self.transformation(sources),
            description=feature.description,
            is_target=False,
            tags=feature.tags,
            constraints=feature.constraints,
        )


class DillTransformationFactory(TransformationFactory):
    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        return CustomTransformationV2.with_method(method=self.method, dtype=self._dtype)

    @abstractproperty
    def method(self) -> Callable[[DataFrame], Series]:
        pass


class CategoricalEncodable(FeatureReferancable):
    def one_hot_encode(self, labels: list[Any]) -> list['Equals']:
        return [self == label for label in labels]

    def label_encoding(self, labels: list[str]) -> 'LabelEncoding':
        return LabelEncoding({label: index for index, label in enumerate(labels)}, self)


class StringTransformable(FeatureReferancable):
    def to_numerical(self) -> 'ToNumerical':
        return ToNumerical(self)


class CustomTransformation(DillTransformationFactory, Transformable):
    def __init__(
        self,
        using_features: list[FeatureReferancable],
        feature: FeatureReferancable,
        transformation: Callable[[DataFrame], Series],
    ) -> None:
        self.using_features = using_features
        self.feature = feature
        self._method = transformation

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        return self._method


class NumericalTransformable(FeatureReferancable):
    def log1p(self) -> 'LogTransform':
        return LogTransform(self)

    def standard_scaled(self) -> 'StandardScalingTransformationFactory':
        return StandardScalingTransformationFactory(self, using_features=[self])


class Float(FeatureFactory, NumericalTransformable):
    _dtype = FeatureType('').float

    def __init__(self, labels: dict[str, str] | None = None):
        self._labels = labels


class Double(FeatureFactory, NumericalTransformable):
    _dtype = FeatureType('').double


class Int32(FeatureFactory, NumericalTransformable):
    _dtype = FeatureType('').int32


class Int64(FeatureFactory, NumericalTransformable):
    _dtype = FeatureType('').int64


class String(FeatureFactory, CategoricalEncodable):
    _dtype = FeatureType('').string

    def split(self, pattern: str, max_splits: int | None = None) -> 'Split':
        return Split(pattern, self, max_splits)

    def replace(self, values: dict[str, str]) -> 'Replace':
        return Replace(values, self)

    def contains(self, value: str) -> 'Contains':
        return Contains(value, self)

    def ordinal_categories(self, orders: list[str]) -> 'Ordinal':
        return Ordinal(orders, self)


class UUID(FeatureFactory):
    _dtype = FeatureType('').uuid


class Bool(FeatureFactory):
    _dtype = FeatureType('').bool


class Array(FeatureFactory):
    _dtype = FeatureType('').array


class Entity(FeatureFactory):

    dtype: FeatureReferancable

    def __init__(self, dtype: FeatureReferancable):
        self.dtype = dtype

    @property
    def _dtype(self) -> FeatureType:
        return self.dtype._dtype


class CreatedAtTimestamp(FeatureFactory):
    _dtype = FeatureType('').datetime


class EventTimestamp(FeatureFactory):

    max_join_with: timedelta | None

    def __init__(self, max_join_with: timedelta | None = None):
        self.max_join_with = max_join_with

    _dtype = FeatureType('').datetime

    def event_timestamp_feature(self, name: str) -> EventTimestampFeature:
        self._name = name
        return EventTimestampFeature(
            name=name,
            ttl=self.max_join_with.seconds if self.max_join_with else None,
            description=self._description,
            tags=self._labels,
        )


class Ratio(TransformationFactory):

    numerator: FeatureReferancable
    denumerator: FeatureReferancable

    def __init__(self, numerator: FeatureReferancable, denumerator: FeatureReferancable) -> None:
        self.using_features = [numerator, denumerator]
        self.feature = Float()
        self.numerator = numerator
        self.denumerator = denumerator

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Ratio as RatioTransformation

        return RatioTransformation(self.numerator.name, self.denumerator.name)


class Ordinal(TransformationFactory):

    orders: list[str]
    in_feature: FeatureReferancable

    def __init__(self, orders: list[str], in_feature: FeatureReferancable) -> None:
        self.using_features = [in_feature]
        self.feature = Int32()
        self.orders = orders
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Ordinal as OrdinalTransformation

        return OrdinalTransformation(self.in_feature.name, self.orders)


class Contains(TransformationFactory):

    text: str
    in_feature: FeatureReferancable

    def __init__(self, text: str, in_feature: FeatureReferancable) -> None:
        self.using_features = [in_feature]
        self.feature = Bool()
        self.in_feature = in_feature
        self.text = text

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Contains as ContainsTransformation

        return ContainsTransformation(self.in_feature.name, self.text)


class Equals(TransformationFactory):

    value: Any
    in_feature: FeatureReferancable

    def __init__(self, value: Any, in_feature: FeatureReferancable) -> None:
        self.using_features = [in_feature]
        self.feature = Bool()
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Equals as EqualsTransformation

        return EqualsTransformation(self.in_feature.name, self.value)


class LabelEncoding(DillTransformationFactory):

    encodings: dict[str, int]
    in_feature: FeatureReferancable

    def __init__(self, labels: dict[str, int], in_feature: FeatureReferancable) -> None:
        self.using_features = [in_feature]
        self.feature = Int32()
        self.in_feature = in_feature
        self.encodings = labels

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def label_encoding(df: DataFrame) -> Series:
            return df[self.in_feature.name].map(self.encodings)

        return label_encoding


class NotEquals(TransformationFactory):

    value: Any
    in_feature: FeatureReferancable

    def __init__(self, value: Any, in_feature: FeatureReferancable) -> None:
        self.using_features = [in_feature]
        self.feature = Bool()
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import NotEquals as NotEqualsTransformation

        return NotEqualsTransformation(self.in_feature.name, self.value)


class GreaterThen(TransformationFactory):

    value: Any
    in_feature: FeatureReferancable

    def __init__(self, value: Any, in_feature: FeatureReferancable) -> None:
        self.using_features = [in_feature]
        self.feature = Bool()
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import GreaterThen as GreaterThenTransformation

        return GreaterThenTransformation(self.in_feature.name, self.value)


class GreaterThenOrEqual(TransformationFactory):

    value: Any
    in_feature: FeatureReferancable

    def __init__(self, value: Any, in_feature: FeatureReferancable) -> None:
        self.using_features = [in_feature]
        self.feature = Bool()
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import GreaterThenOrEqual as GTETransformation

        return GTETransformation(self.in_feature.name, self.value)


class LowerThen(TransformationFactory):

    value: float
    in_feature: FeatureReferancable

    def __init__(self, value: float, in_feature: FeatureReferancable) -> None:
        self.using_features = [in_feature]
        self.feature = Bool()
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import LowerThen as LTTransformation

        return LTTransformation(self.in_feature.name, self.value)


class LowerThenOrEqual(TransformationFactory):

    value: float
    in_feature: FeatureReferancable

    def __init__(self, value: float, in_feature: FeatureReferancable) -> None:
        self.using_features = [in_feature]
        self.feature = Bool()
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import LowerThenOrEqual as LTETransformation

        return LTETransformation(self.in_feature.name, self.value)


class Split(DillTransformationFactory):

    pattern: str
    from_feature: FeatureReferancable
    max_splits: int | None

    def __init__(self, pattern: str, feature: FeatureReferancable, max_splits: int | None = None) -> None:
        self.using_features = [feature]
        self.feature = Array()
        self.pattern = pattern
        self.max_splits = max_splits
        self.from_feature = feature

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def met(df: DataFrame) -> Series:
            return df[self.from_feature.name].str.split(pat=self.pattern, n=self.max_splits)

        return met

    def index(self, index: int) -> 'ArrayIndex':
        return ArrayIndex(index, self)


class ArrayIndex(DillTransformationFactory):

    index: int
    from_feature: FeatureReferancable

    def __init__(self, index: int, feature: FeatureReferancable) -> None:
        self.using_features = [feature]
        self.feature = Bool()
        self.index = index
        self.from_feature = feature

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def met(df: DataFrame) -> Series:
            return df[self.from_feature.name].str[self.index]

        return met


class DateComponent(TransformationFactory):

    component: str
    from_feature: FeatureReferancable

    def __init__(self, component: str, from_feature: FeatureReferancable) -> None:
        self.using_features = [from_feature]
        self.feature = Int32()
        self.from_feature = from_feature
        self.component = component

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import DateComponent as DCTransformation

        return DCTransformation(self.from_feature.name, self.component)


class DifferanceBetween(TransformationFactory):

    first_feature: FeatureReferancable
    second_feature: FeatureReferancable

    def __init__(self, first_feature: FeatureReferancable, second_feature: FeatureReferancable) -> None:
        self.using_features = [first_feature, second_feature]
        self.feature = Float()
        self.first_feature = first_feature
        self.second_feature = second_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Subtraction

        return Subtraction(self.first_feature.name, self.second_feature.name)


class AdditionBetween(TransformationFactory):

    first_feature: FeatureReferancable
    second_feature: FeatureReferancable

    def __init__(self, first_feature: FeatureReferancable, second_feature: FeatureReferancable) -> None:
        self.using_features = [first_feature, second_feature]
        self.feature = Float()
        self.first_feature = first_feature
        self.second_feature = second_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Addition

        return Addition(self.first_feature.name, self.second_feature.name)


class TimeDifferance(TransformationFactory):

    first_feature: FeatureReferancable
    second_feature: FeatureReferancable

    def __init__(self, first_feature: FeatureReferancable, second_feature: FeatureReferancable) -> None:
        self.using_features = [first_feature, second_feature]
        self.feature = Float()
        self.first_feature = first_feature
        self.second_feature = second_feature
        test = FeatureType(name='').datetime
        assert first_feature._dtype == test
        assert second_feature._dtype == test

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import TimeDifference as TDTransformation

        return TDTransformation(self.first_feature.name, self.second_feature.name)


class LogTransform(TransformationFactory):

    source_feature: FeatureReferancable

    def __init__(self, feature: FeatureReferancable) -> None:
        self.using_features = [feature]
        self.feature = Float()
        self.source_feature = feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import LogarithmOnePluss

        return LogarithmOnePluss(self.source_feature.name)


@dataclass
class TimeSeriesTransformationFactory(TransformationFactory):

    field: FeatureReferancable
    agg_method: str
    using_features: list[FeatureReferancable]
    feature: FeatureReferancable = Int64()

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        raise NotImplementedError()
        # from aladdin.psql.data_source import PostgreSQLDataSource

        # assert self.name is not None

        # if len(sources) != 1:
        #     raise ValueError('Expected one source')

        # metadata, request = sources[0]

        # et = request.event_timestamp  # Veeeeery hacky
        # etf = EventTimestamp(timedelta(seconds=et.ttl))
        # etf._name = et.name
        # etf._feature_view = request.feature_view_name
        # self.using_features = [self.field, etf]

        # source = metadata.batch_source
        # if not isinstance(source, PostgreSQLDataSource):
        #     raise ValueError('Only PostgreSQLDataSource is supported')

        # return TimeSeriesTransformation(
        #     method=self.agg_method,
        #     field_name=self.field.name,
        #     table_name=source.table,
        #     config=source.config,
        #     event_timestamp_column=request.event_timestamp.name,
        # )


class Replace(TransformationFactory, StringTransformable):

    values: dict[str, str]
    source_feature: FeatureReferancable

    def __init__(self, values: dict[str, str], feature: FeatureReferancable) -> None:
        self.using_features = [feature]
        self.feature = String()
        self.source_feature = feature
        self.values = values

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import ReplaceStrings

        return ReplaceStrings(self.source_feature.name, self.values)


class ToNumerical(TransformationFactory):

    from_feature: FeatureReferancable

    def __init__(self, feature: FeatureReferancable) -> None:
        self.using_features = [feature]
        self.feature = Float()
        self.from_feature = feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import ToNumerical as ToNumericalTransformation

        return ToNumericalTransformation(self.from_feature.name)


@dataclass
class StandardScalingTransformationFactory(TransformationFactory):

    field: FeatureReferancable

    using_features: list[FeatureReferancable]

    feature: FeatureReferancable = Float()

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        import asyncio

        import nest_asyncio

        from aladdin.enricher import StatisticEricher

        if self.field.is_derived:
            raise ValueError('Standard scaling is not supported for derived features yet')

        assert self.name is not None

        if len(sources) != 1:
            raise ValueError('Expected one source')

        metadata, _ = sources[0]

        if not isinstance(metadata.batch_source, StatisticEricher):
            raise ValueError('The data source needs to conform to StatisticEricher')

        std_enricher = metadata.batch_source.std(columns={self.field.name})
        mean_enricher = metadata.batch_source.mean(columns={self.field.name})

        async def compute() -> tuple[DataFrame, DataFrame]:
            return await asyncio.gather(std_enricher.load(), mean_enricher.load())

        try:
            std, mean = asyncio.get_event_loop().run_until_complete(compute())
        except RuntimeError:
            nest_asyncio.apply()
            std, mean = asyncio.get_event_loop().run_until_complete(compute())

        return StandardScalingTransformation(mean[self.field.name], std[self.field.name], self.field.name)


class IsIn(TransformationFactory):

    field: FeatureReferancable
    values: list

    def __init__(self, feature: FeatureReferancable, values: list) -> None:
        self.using_features = [feature]
        self.values = values
        self.feature = Bool()
        self.field = feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import IsIn as IsInTransformation

        return IsInTransformation(self.values, self.field.name)


class And(TransformationFactory):

    first_feature: FeatureReferancable
    other_feature: FeatureReferancable

    def __init__(self, feature: FeatureReferancable, other_feature: FeatureReferancable) -> None:
        self.using_features = [feature, other_feature]
        self.feature = Bool()
        self.first_feature = feature
        self.other_feature = other_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import And as AndTransformation

        return AndTransformation(self.first_feature.name, self.other_feature.name)


class Or(TransformationFactory):

    first_feature: FeatureReferancable
    other_feature: FeatureReferancable

    def __init__(self, feature: FeatureReferancable, other_feature: FeatureReferancable) -> None:
        self.using_features = [feature, other_feature]
        self.feature = Bool()
        self.first_feature = feature
        self.other_feature = other_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Or as OrTransformation

        return OrTransformation(self.first_feature.name, self.other_feature.name)


class Inverse(TransformationFactory):

    from_feature: FeatureReferancable

    def __init__(self, feature: FeatureReferancable) -> None:
        self.using_features = [feature]
        self.feature = Bool()
        self.from_feature = feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Inverse as InverseTransformation

        return InverseTransformation(self.from_feature.name)


class FillMissing(TransformationFactory):

    from_feature: FeatureReferancable
    strategy: Literal['mean', 'median']

    def __init__(self, feature: FeatureReferancable, strategy: Literal['mean', 'median']) -> None:
        self.from_feature = feature
        self.using_features = [feature]
        self.feature = {'string': String(), 'int32': Int32(), 'int64': Int64(), 'float': Float()}[
            feature._dtype.name
        ]
        self.strategy = strategy

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import FillMissingTransformation

        return FillMissingTransformation(
            key=self.from_feature.name, strategy=self.strategy, dtype=self.from_feature._dtype
        )
