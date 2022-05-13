from abc import ABC, abstractproperty
from types import MethodType
from typing import Any, Optional, Callable
from pandas import DataFrame, Series # type: ignore
from dataclasses import dataclass
from datetime import timedelta
import numpy as np

from aladdin.feature import Constraint, Feature, FeatureType, Above
from aladdin.transformation import Transformation


@dataclass
class FeatureReference:
    feature_view: str
    feature: str


class FeatureFactory(ABC):

    labels: dict[str, str] | None = None
    name: str | None = None
    feature_view: str | None = None
    constraints: set[Constraint] = set()

    @abstractproperty
    def _dtype(self) -> type:
        pass

    def feature(self, name: str) -> Feature:
        self.name = name
        return Feature(name, dtype=self._dtype, description=None, is_target=False, tags=self.labels, constraints=self.constraints)

    def transformed(self, using_features: list[str], transformation: Callable[[DataFrame], Series]) -> "TransformationFactory":
        return CustomTransformation(using_features, self, transformation)
    

@dataclass
class CompiledTransformation:
    using_features: list[FeatureReference]
    feature: Feature
    transformation: MethodType

    def transform(self, df: DataFrame) -> DataFrame:
        df[self.feature.name] = self.transformation(df)
        return df


@dataclass
class CustomTransformationV2(Transformation):

    method: bytes
    dtype: FeatureType | None = None # Should be something else
    name: str = "custom_transformation"

    @staticmethod
    def with_method(method: Callable[[DataFrame], Series]) -> "CustomTransformationV2":
        import dill
        return CustomTransformationV2(method=dill.dumps(method))

    async def transform(self, df: DataFrame) -> Series:
        import dill
        loaded = dill.loads(self.method)
        return await loaded(df)



class TransformationFactory(ABC):

    using_features: list[FeatureFactory]
    feature: FeatureFactory
    name: str | None

    @property
    def transformation(self) -> Transformation:
        return CustomTransformationV2.with_method(method=self.method)

    @property
    def using_feature_names(self) -> list[str]:
        return [feature.name if isinstance(feature, FeatureFactory) else feature for feature in self.using_features]

    def __init__(self, using_features: list[str | FeatureFactory], feature: FeatureFactory) -> None:
        self.using_features = using_features
        self.feature = feature

    @abstractproperty
    def method(self) -> Callable[[DataFrame], Series]:
        pass

class CustomTransformation(TransformationFactory):

    def __init__(self, using_features: list[str | FeatureFactory], feature: FeatureFactory, transformation: Callable[[DataFrame], Series]) -> None:
        self.using_features = using_features
        self.feature = feature
        self._method = transformation

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        return self._method

class Float(FeatureFactory):
    _dtype = FeatureType.float

    def __init__(self, labels: Optional[dict[str, str]] = None):
        self.labels = labels

    def log1p(self, using: str) -> "LogTransform":
        return LogTransform(using, self.labels)

class Double(FeatureFactory):
    _dtype = FeatureType.double

    def __init__(self, labels: Optional[dict[str, str]] = None):
        self.labels = labels

class Int32(FeatureFactory):
    _dtype = FeatureType.int32

    def __init__(self, labels: Optional[dict[str, str]] = None):
        self.labels = labels

class Int64(FeatureFactory):
    _dtype = FeatureType.int64

    def __init__(self, labels: Optional[dict[str, str]] = None):
        self.labels = labels

class String(FeatureFactory):
    _dtype = FeatureType.string

    def __init__(self, labels: Optional[dict[str, str]] = None):
        self.labels = labels

class UUID(FeatureFactory):
    _dtype = FeatureType.uuid

    def __init__(self, labels: Optional[dict[str, str]] = None):
        self.labels = labels

class Bool(FeatureFactory):
    _dtype = FeatureType.bool

    def __init__(self, labels: Optional[dict[str, str]] = None):
        self.labels = labels

class Entity(FeatureFactory):

    dtype: FeatureFactory
    description: str
    
    def __init__(self, dtype: FeatureFactory, description: str | None = None, labels: Optional[dict[str, str]] = None):
        self.dtype = dtype
        self.labels = labels
        self.description = description or ""

    @property
    def _dtype(self) -> type:
        return self.dtype._dtype


class CreatedAtTimestamp(FeatureFactory):
    _dtype = FeatureType.datetime

class EventTimestamp(FeatureFactory):

    max_join_with: timedelta

    def __init__(self, max_join_with: timedelta):
        self.max_join_with = max_join_with

    _dtype = FeatureType.datetime


class Ratio(TransformationFactory):

    numerator: FeatureFactory
    denumerator: FeatureFactory

    def __init__(self, numerator: FeatureFactory, denumerator: FeatureFactory, labels: dict[str, str] | None = None) -> None:
        super().__init__([numerator, denumerator], Float(labels))
        self.numerator = numerator
        self.denumerator = denumerator

    @staticmethod
    def ratio(numerator: str, denumirator: str, df: DataFrame) -> Series:
        from numpy import nan
        mask = df[numerator].isna() | df[denumirator].isna() | df[denumirator] == 0
        results = df[numerator].copy()
        results.loc[mask] = nan
        results.loc[~mask] = df.loc[~mask, numerator].astype(float) / df.loc[~mask, denumirator].astype(float)
        return results

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def ratio(df: DataFrame) -> Series:
            return Ratio.ratio(self.numerator.name, self.denumerator.name, df)
        return ratio

class Contains(TransformationFactory):

    text: str
    in_feature: FeatureFactory

    def __init__(self, text: str, in_feature: FeatureFactory, labels: dict[str, str] | None = None) -> None:
        super().__init__([in_feature], Bool(labels))
        self.in_feature = in_feature
        self.text = text


    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def contains(df: DataFrame) -> Series:
            return df[self.in_feature.name].str.contains(self.text)
        
        return contains

class Equals(TransformationFactory):
    
    value: Any
    in_feature: FeatureFactory

    def __init__(self, value: Any, in_feature: str, labels: dict[str, str] | None = None) -> None:
        super().__init__([in_feature], Bool(labels))
        self.value = value
        self.in_feature = in_feature

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def equals(df: DataFrame) -> Series:
            return df[self.in_feature.name] == self.value
        return equals

class DateComponent(TransformationFactory):

    component: str
    from_feature: FeatureFactory

    def __init__(self, component: str, from_feature: FeatureFactory, labels: dict[str, str] | None = None) -> None:
        super().__init__([from_feature], Int32(labels))
        self.from_feature = from_feature
        self.component = component

    @staticmethod
    def date_component(component: str, feature: str, df: DataFrame) -> Series:
        from pandas import to_datetime
        from numpy import nan
        mask = df[feature].isna()
        results = df[feature].copy()
        results.loc[mask] = nan
        results.loc[~mask] = getattr(to_datetime(df.loc[~mask, feature]).dt, component)
        return results

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def date_component(df: DataFrame) -> Series:
            return DateComponent.date_component(self.component, self.from_feature.name, df)
        return date_component

class DifferanceBetween(TransformationFactory):
    
    first_feature: FeatureFactory
    second_feature: FeatureFactory

    def __init__(self, first_feature: FeatureFactory, second_feature: FeatureFactory, labels: dict[str, str] | None = None) -> None:
        super().__init__([first_feature, second_feature], Float(labels))
        self.first_feature = first_feature
        self.second_feature = second_feature

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def differance_between(df: DataFrame) -> Series:
            return df[self.first_feature.name] - df[self.second_feature.name]
        return differance_between


class TimeDifferance(TransformationFactory):

    first_feature: FeatureFactory
    second_feature: FeatureFactory

    def __init__(self, first_feature: FeatureFactory, second_feature: FeatureFactory, labels: dict[str, str] | None = None) -> None:
        super().__init__([first_feature, second_feature], Float(labels))
        self.first_feature = first_feature
        self.second_feature = second_feature
        test = FeatureType(name="").datetime
        assert first_feature._dtype == test
        assert second_feature._dtype == test
        
    @staticmethod
    def time_differance(first_feature: str, second_feature: str, df: DataFrame) -> Series:
        mask = df[first_feature].isna() | df[second_feature].isna()
        results = df[first_feature].copy()
        results.loc[mask] = np.nan
        results.loc[~mask] = (df.loc[~mask, first_feature] - df.loc[~mask, second_feature]) / np.timedelta64(1, 's')
        return results

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def time_differance(df: DataFrame) -> Series:
            return TimeDifferance.time_differance(self.first_feature.name, self.second_feature.name, df)
        return time_differance

class LogTransform(TransformationFactory):

    feature: FeatureFactory

    def __init__(self, feature: FeatureFactory, labels: dict[str, str] | None = None) -> None:
        super().__init__([feature], Float(labels))
        self.feature = feature

    @staticmethod
    def log_transform(feature: str, df: DataFrame) -> Series:
        from numpy import nan
        mask = df[feature].isna()
        results = df[feature].copy()
        results.loc[mask] = nan
        results.loc[~mask] = np.log1p(df.loc[~mask, feature])
        return results

    @property
    def method(self) -> Callable[[DataFrame], Series]:
        async def log_transform(df: DataFrame) -> Series:
            return LogTransform.log_transform(self.feature.name, df)
        return log_transform