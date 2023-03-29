from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aligned.schemas.codable import Codable
from aligned.schemas.constraints import Constraint


@dataclass
class FeatureType(Codable):
    # FIXME: Should use a more Pythonic design, as this one did not behave as intended

    name: str

    @property
    def is_numeric(self) -> bool:
        return self.name in {'bool', 'int32', 'int64', 'float', 'double'}  # Can be represented as an int

    @property
    def python_type(self) -> type:
        from datetime import date, datetime, time, timedelta
        from uuid import UUID

        from numpy import double

        return {
            'string': str,
            'int32': int,
            'int64': int,
            'float': float,
            'double': double,
            'bool': bool,
            'date': date,
            'datetime': datetime,
            'time': time,
            'timedelta': timedelta,
            'uuid': UUID,
            'array': list,
            'embedding': list,
        }[self.name]

    @property
    def pandas_type(self) -> str | type:
        import numpy as np

        return {
            'string': str,
            'int32': 'Int32',
            'int64': 'Int64',
            'float': np.float64,
            'double': np.double,
            'bool': 'boolean',
            'date': np.datetime64,
            'datetime': np.datetime64,
            'time': np.datetime64,
            'timedelta': np.timedelta64,
            'uuid': str,
            'array': list,
            'embedding': list,
        }[self.name]

    @property
    def polars_type(self) -> type:
        import polars as pl

        return {
            'string': pl.Utf8,
            'int32': pl.Int32,
            'int64': pl.Int64,
            'float': pl.Float64,
            'double': pl.Float64,
            'bool': pl.Boolean,
            'date': pl.Date,
            'datetime': pl.Datetime,
            'time': pl.Time,
            'timedelta': pl.Duration,
            'uuid': pl.Utf8,
            'array': pl.List,
            'embedding': pl.List,
        }[self.name]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FeatureType):
            return self.name == other.name
        return False

    def __hash__(self) -> int:
        return self.name.__hash__()

    def __pre_serialize__(self) -> FeatureType:
        assert isinstance(self.name, str)
        return self

    @property
    def string(self) -> FeatureType:
        return FeatureType(name='string')

    @property
    def int32(self) -> FeatureType:
        return FeatureType(name='int32')

    @property
    def bool(self) -> FeatureType:
        return FeatureType(name='bool')

    @property
    def int64(self) -> FeatureType:
        return FeatureType(name='int64')

    @property
    def float(self) -> FeatureType:
        return FeatureType(name='float')

    @property
    def double(self) -> FeatureType:
        return FeatureType(name='double')

    @property
    def date(self) -> FeatureType:
        return FeatureType(name='date')

    @property
    def uuid(self) -> FeatureType:
        return FeatureType(name='uuid')

    @property
    def datetime(self) -> FeatureType:
        return FeatureType(name='datetime')

    @property
    def array(self) -> FeatureType:
        return FeatureType(name='array')

    @property
    def embedding(self) -> FeatureType:
        return FeatureType(name='embedding')


@dataclass
class Feature(Codable):
    name: str
    dtype: FeatureType
    description: str | None = None
    tags: dict[str, str] | None = None

    constraints: set[Constraint] | None = None

    def __pre_serialize__(self) -> Feature:
        assert isinstance(self.name, str)
        assert isinstance(self.dtype, FeatureType)
        assert isinstance(self.description, str) or self.description is None
        assert isinstance(self.tags, dict) or self.tags is None
        if self.constraints:
            for constraint in self.constraints:
                assert isinstance(constraint, Constraint)

        return self

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        value = f'{self.name} - {self.dtype.name}'
        if self.description:
            value += f' - {self.description}'
        return value


@dataclass
class EventTimestamp(Codable):
    name: str
    ttl: int | None = None
    description: str | None = None
    tags: dict[str, str] | None = None
    dtype: FeatureType = FeatureType('').datetime

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        value = f'{self.name} - {self.dtype.name}'
        if self.description:
            value += f' - {self.description}'
        return value


@dataclass
class FeatureLocation(Codable):
    name: str
    location: Literal['feature_view', 'combined_view', 'model']

    @property
    def identifier(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f'{self.location}:{self.name}'

    def __hash__(self) -> int:
        return (self.name + self.location).__hash__()

    @staticmethod
    def feature_view(name: str) -> FeatureLocation:
        return FeatureLocation(name, 'feature_view')

    @staticmethod
    def combined_view(name: str) -> FeatureLocation:
        return FeatureLocation(name, 'combined_view')

    @staticmethod
    def model(name: str) -> FeatureLocation:
        return FeatureLocation(name, 'model')


@dataclass
class FeatureReferance(Codable):
    name: str
    location: FeatureLocation
    dtype: FeatureType
    # is_derived: bool

    def __hash__(self) -> int:
        return hash(self.name)
