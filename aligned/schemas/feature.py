from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING
from zoneinfo import ZoneInfo

import polars as pl

from aligned.schemas.codable import Codable
from aligned.schemas.constraints import Constraint

if TYPE_CHECKING:
    from aligned.compiler.feature_factory import FeatureFactory


NAME_POLARS_MAPPING = [
    ('string', pl.Utf8),
    ('int8', pl.Int8),
    ('int16', pl.Int16),
    ('int32', pl.Int32),
    ('int64', pl.Int64),
    ('uint8', pl.UInt8),
    ('uint16', pl.UInt16),
    ('uint32', pl.UInt32),
    ('uint64', pl.UInt64),
    ('float', pl.Float64),
    ('float', pl.Float32),
    ('double', pl.Float64),
    ('bool', pl.Boolean),
    ('date', pl.Date),
    ('datetime', pl.Datetime),
    ('time', pl.Time),
    ('timedelta', pl.Duration),
    ('uuid', pl.Utf8),
    ('array', pl.List(pl.Utf8)),
    ('embedding', pl.List),
    ('json', pl.Utf8),
]


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
            'uint8',
            'uint16',
            'uint32',
            'uint64',
            'float',
            'double',
        }  # Can be represented as an int

    @property
    def is_datetime(self) -> bool:
        return self.name.startswith('datetime')

    @property
    def is_array(self) -> bool:
        return self.name.startswith('array')

    def array_subtype(self) -> FeatureType | None:
        if not self.is_array or '-' not in self.name:
            return None

        sub = str(self.name[len('array-') :])
        return FeatureType(sub)

    @property
    def datetime_timezone(self) -> str | None:
        if not self.is_datetime:
            return None

        return self.name.split('-')[1] if '-' in self.name else None

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
        if self.is_datetime:
            time_zone = self.datetime_timezone
            return pl.Datetime(time_zone=time_zone)  # type: ignore

        if self.is_array:
            sub_type = self.array_subtype()
            if sub_type:
                return pl.List(sub_type.polars_type)  # type: ignore
            else:
                return pl.List(pl.Utf8)  # type: ignore

        for name, dtype in NAME_POLARS_MAPPING:
            if name == self.name:
                return dtype

        raise ValueError(f'Unable to find a value that can represent {self.name}')

    @property
    def feature_factory(self) -> FeatureFactory:
        from aligned.compiler import feature_factory as ff

        if self.name.startswith('datetime-'):
            time_zone = self.name.split('-')[1]
            return ff.Timestamp(time_zone=time_zone)

        if self.name.startswith('array-'):
            sub_type = '-'.join(self.name.split('-')[1:])
            return ff.List(FeatureType(name=sub_type).feature_factory)

        return {
            'string': ff.String(),
            'int8': ff.Int8(),
            'int16': ff.Int16(),
            'int32': ff.Int32(),
            'int64': ff.Int64(),
            'uint8': ff.UInt8(),
            'uint16': ff.UInt16(),
            'uint32': ff.UInt32(),
            'uint64': ff.UInt64(),
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
        if isinstance(polars_type, pl.Datetime):
            if polars_type.time_zone:
                return FeatureType(name=f'datetime-{polars_type.time_zone}')
            return FeatureType(name='datetime')

        if isinstance(polars_type, pl.List):
            if polars_type.inner:
                sub_type = FeatureType.from_polars(polars_type.inner)  # type: ignore
                return FeatureType(name=f'array-{sub_type.name}')

            return FeatureType(name='array')

        if isinstance(polars_type, pl.Struct):
            return FeatureType(name='json')

        for name, dtype in NAME_POLARS_MAPPING:
            if polars_type.is_(dtype):
                return FeatureType(name=name)

        raise ValueError(f'Unable to find a value that can represent {polars_type}')

    @staticmethod
    def string() -> FeatureType:
        return FeatureType(name='string')

    @staticmethod
    def uint8() -> FeatureType:
        return FeatureType(name='uint8')

    @staticmethod
    def uint16() -> FeatureType:
        return FeatureType(name='uint16')

    @staticmethod
    def uint32() -> FeatureType:
        return FeatureType(name='uint32')

    @staticmethod
    def uint64() -> FeatureType:
        return FeatureType(name='uint64')

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
    def datetime(tz: ZoneInfo | None = ZoneInfo('UTC')) -> FeatureType:
        if not tz:
            return FeatureType(name='datetime')
        return FeatureType(name=f'datetime-{tz.key}')

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

    def as_reference(self, location: FeatureLocation) -> FeatureReference:
        return FeatureReference(
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
    dtype: FeatureType = field(default_factory=lambda: FeatureType.datetime())

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        value = f'{self.name} - {self.dtype.name}'
        if self.description:
            value += f' - {self.description}'
        return value

    def as_feature(self) -> Feature:
        return Feature(
            name=self.name,
            dtype=self.dtype,
            description=self.description,
            tags=self.tags,
        )


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
class FeatureReference(Codable):
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

    def feature_reference(self) -> FeatureReference:
        return self
