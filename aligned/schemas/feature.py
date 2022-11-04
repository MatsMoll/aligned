from dataclasses import dataclass

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
        }[self.name]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FeatureType):
            return self.name == other.name
        return False

    @property
    def string(self) -> 'FeatureType':
        return FeatureType(name='string')

    @property
    def int32(self) -> 'FeatureType':
        return FeatureType(name='int32')

    @property
    def bool(self) -> 'FeatureType':
        return FeatureType(name='bool')

    @property
    def int64(self) -> 'FeatureType':
        return FeatureType(name='int64')

    @property
    def float(self) -> 'FeatureType':
        return FeatureType(name='float')

    @property
    def double(self) -> 'FeatureType':
        return FeatureType(name='double')

    @property
    def date(self) -> 'FeatureType':
        return FeatureType(name='date')

    @property
    def uuid(self) -> 'FeatureType':
        return FeatureType(name='uuid')

    @property
    def datetime(self) -> 'FeatureType':
        return FeatureType(name='datetime')

    @property
    def array(self) -> 'FeatureType':
        return FeatureType(name='array')


@dataclass
class Feature(Codable):
    name: str
    dtype: FeatureType
    description: str | None = None
    tags: dict[str, str] | None = None

    constraints: set[Constraint] | None = None

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
class FeatureReferance(Codable):
    name: str
    feature_view: str
    dtype: FeatureType
    is_derived: bool

    def __hash__(self) -> int:
        return hash(self.name)
