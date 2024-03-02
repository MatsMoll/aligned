from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

import aligned.compiler.feature_factory as ff
from aligned.schemas.codable import Codable
from aligned.schemas.constraints import Constraint

NAME_POLARS_MAPPING = {
    'string': pl.Utf8,
    'int8': pl.Int8,
    'int16': pl.Int16,
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
    'array': pl.List(pl.Utf8),
    'embedding': pl.List,
    'json': pl.Utf8,
}


@dataclass
class FeatureType(Codable):
    # FIXME: Should use a more Pythonic design, as this one did not behave as intended

    name: str

    @property
    def is_numeric(self) -> bool:
        return self.name in {
            'bool',
            'int8',
            'int16',
            'int32',
            'int64',
            'float',
            'double',
        }  # Can be represented as an int

    @property
    def python_type(self) -> type:
        from datetime import date, datetime, time, timedelta
        from uuid import UUID

        from numpy import double

        return {
            'string': str,
            'int8': int,
            'int16': int,
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
            'json': str,
        }[self.name]

    @property
    def pandas_type(self) -> str | type:
        import numpy as np

        return {
            'string': str,
            'int8': 'Int8',
            'int16': 'Int16',
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
            'json': str,
        }[self.name]

    @property
    def polars_type(self) -> type:
        return NAME_POLARS_MAPPING[self.name]

    @property
    def feature_factory(self) -> ff.FeatureFactory:
        return {
            'string': ff.String(),
            'int8': ff.Int8(),
            'int16': ff.Int16(),
            'int32': ff.Int32(),
            'int64': ff.Int64(),
            'float': ff.Float(),
            'double': ff.Float(),
            'bool': ff.Bool(),
            'date': ff.Timestamp(),
            'datetime': ff.Timestamp(),
            'time': ff.Timestamp(),
            'timedelta': ff.Timestamp(),
            'uuid': ff.UUID(),
            'array': ff.Embedding(),
            'embedding': ff.Embedding(),
            'json': ff.Json(),
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

    @staticmethod
    def from_polars(polars_type: pl.DataType) -> FeatureType:
        for name, dtype in NAME_POLARS_MAPPING.items():
            if polars_type.is_(dtype):
                return FeatureType(name=name)
        raise ValueError(f'Unable to find a value that can represent {polars_type}')

    @staticmethod
    def string() -> FeatureType:
        return FeatureType(name='string')

    @staticmethod
    def int8() -> FeatureType:
        return FeatureType(name='int8')

    @staticmethod
    def int16() -> FeatureType:
        return FeatureType(name='int16')

    @staticmethod
    def int32() -> FeatureType:
        return FeatureType(name='int32')

    @staticmethod
    def bool() -> FeatureType:
        return FeatureType(name='bool')

    @staticmethod
    def int64() -> FeatureType:
        return FeatureType(name='int64')

    @staticmethod
    def float() -> FeatureType:
        return FeatureType(name='float')

    @staticmethod
    def double() -> FeatureType:
        return FeatureType(name='double')

    @staticmethod
    def date() -> FeatureType:
        return FeatureType(name='date')

    @staticmethod
    def uuid() -> FeatureType:
        return FeatureType(name='uuid')

    @staticmethod
    def datetime() -> FeatureType:
        return FeatureType(name='datetime')

    @staticmethod
    def json() -> FeatureType:
        return FeatureType(name='json')

    @staticmethod
    def array() -> FeatureType:
        return FeatureType(name='array')

    @staticmethod
    def embedding() -> FeatureType:
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

    def renamed(self, new_name: str) -> Feature:
        return Feature(
            name=new_name,
            dtype=self.dtype,
            description=self.description,
            tags=self.tags,
            constraints=self.constraints,
        )

    def as_reference(self, location: FeatureLocation) -> FeatureReferance:
        return FeatureReferance(
            name=self.name,
            location=location,
            dtype=self.dtype,
        )

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
    dtype: FeatureType = FeatureType.datetime()

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

    @staticmethod
    def from_string(string: str) -> FeatureLocation:
        splits = string.split(':')
        return FeatureLocation(name=splits[1], location=splits[0])


@dataclass
class FeatureReferance(Codable):
    name: str
    location: FeatureLocation
    dtype: FeatureType
    # is_derived: bool

    def as_feature(self) -> Feature:
        return Feature(
            name=self.name,
            dtype=self.dtype,
            description=None,
            tags=None,
            constraints=None,
        )

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def identifier(self) -> str:
        return f'{self.location.identifier}:{self.name}'
