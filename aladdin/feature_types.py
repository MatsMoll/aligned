from abc import ABC, abstractproperty
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, TypeVar

from pandas import DataFrame, Series

from aladdin.feature import Constraint
from aladdin.feature import EventTimestamp as EventTimestampFeature
from aladdin.feature import Feature, FeatureReferance, FeatureType
from aladdin.feature_view.feature_view_metadata import FeatureViewMetadata
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.transformation import StandardScalingTransformation, TimeSeriesTransformation, Transformation


class FeatureReferancable:

    _name: str | None
    _feature_view: str | None

    @property
    def is_derived(self) -> bool:
        return isinstance(self, TransformationFactory)

    @property
    def name(self) -> str:
        if not self._name:
            raise ValueError('Name is not set')
        return self._name

    @property
    def feature_view(self) -> str:
        if not self._feature_view:
            raise ValueError('Feature view is not set')
        return self._feature_view

    @abstractproperty
    def _dtype(self) -> FeatureType:
        pass

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
    ) -> 'TransformationFactory':
        import asyncio

        if asyncio.iscoroutinefunction(transformation):
            return CustomTransformation(using_features or [self], self, transformation)

        async def sub_tran(df: DataFrame) -> Series:
            return transformation(df)

        return CustomTransformation(using_features or [self], self, sub_tran)

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

    def __sub__(self, other: FeatureReferancable) -> 'Transformable':
        if self._dtype == FeatureType('').datetime:
            return TimeDifferance(self, other)
        return DifferanceBetween(self, other)

    def __add__(self, other: FeatureReferancable) -> 'AdditionBetween':
        return AdditionBetween(self, other)


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

    def count(self) -> 'TimeSeriesTransformationFactory':
        return TimeSeriesTransformationFactory(self, 'COUNT', [])

    def sum(self) -> 'TimeSeriesTransformationFactory':
        return TimeSeriesTransformationFactory(self, 'SUM', [])

    def mean(self) -> 'TimeSeriesTransformationFactory':
        return TimeSeriesTransformationFactory(self, 'AVG', [])


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

    @abstractproperty
    def transformation(  # type: ignore
        self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]
    ) -> Transformation:
        pass

    @property
    def using_feature_names(self) -> list[str]:
        return [feature.name for feature in self.using_features]

    def __init__(self, using_features: list[FeatureReferancable], feature: FeatureReferancable) -> None:
        self.using_features = using_features
        self.feature = feature

    def labels(self: FeatureTypeVar, labels: dict[str, str]) -> FeatureTypeVar:
        if isinstance(self, TransformationFactory) and isinstance(self.feature, FeatureFactory):
            self.feature._labels = labels
        return self

    def description(self: FeatureTypeVar, description: str) -> FeatureTypeVar:
        if isinstance(self, TransformationFactory) and isinstance(self.feature, FeatureFactory):
            self.feature._description = description
        return self


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

    max_join_with: timedelta

    def __init__(self, max_join_with: timedelta):
        self.max_join_with = max_join_with

    _dtype = FeatureType('').datetime

    def event_timestamp_feature(self, name: str) -> EventTimestampFeature:
        self._name = name
        return EventTimestampFeature(
            name=name,
            ttl=self.max_join_with.seconds,
            description=self._description,
            tags=self._labels,
        )


class Ratio(TransformationFactory):

    numerator: FeatureReferancable
    denumerator: FeatureReferancable

    def __init__(self, numerator: FeatureReferancable, denumerator: FeatureReferancable) -> None:
        super().__init__([numerator, denumerator], Float())
        self.numerator = numerator
        self.denumerator = denumerator

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Ratio as RatioTransformation

        return RatioTransformation(self.numerator.name, self.denumerator.name)


class Contains(TransformationFactory):

    text: str
    in_feature: FeatureReferancable

    def __init__(self, text: str, in_feature: FeatureReferancable) -> None:
        super().__init__([in_feature], Bool())
        self.in_feature = in_feature
        self.text = text

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Contains as ContainsTransformation

        return ContainsTransformation(self.in_feature.name, self.text)


class Equals(TransformationFactory):

    value: Any
    in_feature: FeatureReferancable

    def __init__(self, value: Any, in_feature: FeatureReferancable) -> None:
        super().__init__([in_feature], Bool())
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Equals as EqualsTransformation

        return EqualsTransformation(self.in_feature.name, self.value)


class LabelEncoding(DillTransformationFactory):

    encodings: dict[str, int]
    in_feature: FeatureReferancable

    def __init__(self, labels: dict[str, int], in_feature: FeatureReferancable) -> None:
        super().__init__([in_feature], Int32())
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
        super().__init__([in_feature], Bool())
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import NotEquals as NotEqualsTransformation

        return NotEqualsTransformation(self.in_feature.name, self.value)


class GreaterThen(TransformationFactory):

    value: Any
    in_feature: FeatureReferancable

    def __init__(self, value: Any, in_feature: FeatureReferancable) -> None:
        super().__init__([in_feature], Bool())
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import GreaterThen as GreaterThenTransformation

        return GreaterThenTransformation(self.in_feature.name, self.value)


class GreaterThenOrEqual(TransformationFactory):

    value: Any
    in_feature: FeatureReferancable

    def __init__(self, value: Any, in_feature: FeatureReferancable) -> None:
        super().__init__([in_feature], Bool())
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import GreaterThenOrEqual as GTETransformation

        return GTETransformation(self.in_feature.name, self.value)


class LowerThen(TransformationFactory):

    value: float
    in_feature: FeatureReferancable

    def __init__(self, value: float, in_feature: FeatureReferancable) -> None:
        super().__init__([in_feature], Bool())
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import LowerThen as LTTransformation

        return LTTransformation(self.in_feature.name, self.value)


class LowerThenOrEqual(TransformationFactory):

    value: float
    in_feature: FeatureReferancable

    def __init__(self, value: float, in_feature: FeatureReferancable) -> None:
        super().__init__([in_feature], Bool())
        self.value = value
        self.in_feature = in_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import LowerThenOrEqual as LTETransformation

        return LTETransformation(self.in_feature.name, self.value)


class Split(DillTransformationFactory):

    pattern: str
    feature: FeatureReferancable
    max_splits: int | None

    def __init__(self, pattern: str, feature: FeatureReferancable, max_splits: int | None = None) -> None:
        super().__init__([feature], Array())
        self.pattern = pattern
        self.max_splits = max_splits
        self.feature = feature

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def met(df: DataFrame) -> Series:
            return df[self.feature.name].str.split(pat=self.pattern, n=self.max_splits)

        return met


class ArrayIndex(DillTransformationFactory):

    index: int
    feature: FeatureReferancable

    def __init__(self, index: int, feature: FeatureReferancable) -> None:
        super().__init__([feature], Bool())
        self.index = index
        self.feature = feature

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def met(df: DataFrame) -> Series:
            return df[self.feature.name].str[self.index]

        return met


class DateComponent(TransformationFactory):

    component: str
    from_feature: FeatureReferancable

    def __init__(self, component: str, from_feature: FeatureReferancable) -> None:
        super().__init__([from_feature], Int32())
        self.from_feature = from_feature
        self.component = component

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import DateComponent as DCTransformation

        return DCTransformation(self.from_feature.name, self.component)


class DifferanceBetween(TransformationFactory):

    first_feature: FeatureReferancable
    second_feature: FeatureReferancable

    def __init__(self, first_feature: FeatureReferancable, second_feature: FeatureReferancable) -> None:
        super().__init__([first_feature, second_feature], Float())
        self.first_feature = first_feature
        self.second_feature = second_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Subtraction

        return Subtraction(self.first_feature.name, self.second_feature.name)


class AdditionBetween(TransformationFactory):

    first_feature: FeatureReferancable
    second_feature: FeatureReferancable

    def __init__(self, first_feature: FeatureReferancable, second_feature: FeatureReferancable) -> None:
        super().__init__([first_feature, second_feature], Float())
        self.first_feature = first_feature
        self.second_feature = second_feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import Addition

        return Addition(self.first_feature.name, self.second_feature.name)


class TimeDifferance(TransformationFactory):

    first_feature: FeatureReferancable
    second_feature: FeatureReferancable

    def __init__(self, first_feature: FeatureReferancable, second_feature: FeatureReferancable) -> None:
        super().__init__([first_feature, second_feature], Float())
        self.first_feature = first_feature
        self.second_feature = second_feature
        test = FeatureType(name='').datetime
        assert first_feature._dtype == test
        assert second_feature._dtype == test

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import TimeDifference as TDTransformation

        return TDTransformation(self.first_feature.name, self.second_feature.name)


class LogTransform(TransformationFactory):

    feature: FeatureReferancable

    def __init__(self, feature: FeatureReferancable) -> None:
        super().__init__([feature], Float())
        self.feature = feature

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.transformation import LogarithmOnePluss

        return LogarithmOnePluss(self.feature.name)


@dataclass
class TimeSeriesTransformationFactory(TransformationFactory):

    field: FeatureReferancable
    agg_method: str
    using_features: list[FeatureReferancable]
    feature: FeatureReferancable = Int64()

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        from aladdin.psql.data_source import PostgreSQLDataSource

        assert self.name is not None

        if len(sources) != 1:
            raise ValueError('Expected one source')

        metadata, request = sources[0]

        et = request.event_timestamp  # Veeeeery hacky
        etf = EventTimestamp(timedelta(seconds=et.ttl))
        etf._name = et.name
        etf._feature_view = request.feature_view_name
        self.using_features = [self.field, etf]

        source = metadata.batch_source
        if not isinstance(source, PostgreSQLDataSource):
            raise ValueError('Only PostgreSQLDataSource is supported')

        return TimeSeriesTransformation(
            method=self.agg_method,
            field_name=self.field.name,
            table_name=source.table,
            config=source.config,
            event_timestamp_column=request.event_timestamp.name,
        )


@dataclass
class StandardScalingTransformationFactory(TransformationFactory):

    field: FeatureReferancable

    using_features: list[FeatureReferancable]

    feature: FeatureReferancable = Float()

    def transformation(self, sources: list[tuple[FeatureViewMetadata, RetrivalRequest]]) -> Transformation:
        import asyncio

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

        std, mean = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(std_enricher.load(), mean_enricher.load())
        )
        return StandardScalingTransformation(mean, std, self.field.name)
