from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable
from aligned.schemas.feature import FeatureType


class SupportedLiteralValues:

    values: dict[str, type[LiteralValue]]

    _shared: SupportedLiteralValues | None = None

    def __init__(self) -> None:
        self.values = {}
        for lit in [IntValue, FloatValue, BoolValue, DateValue, DatetimeValue, StringValue, ArrayValue]:
            self.values[lit.name] = lit

    @classmethod
    def shared(cls) -> SupportedLiteralValues:
        if cls._shared:
            return cls._shared
        cls._shared = SupportedLiteralValues()
        return cls._shared


class LiteralValue(Codable, SerializableType):
    name: str

    @property
    def python_value(self) -> Any:
        raise NotImplementedError()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType(self.name)

    def _serialize(self) -> dict:
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> LiteralValue:
        name_type = value['name']
        del value['name']
        data_class = SupportedLiteralValues.shared().values[name_type]
        return data_class.from_dict(value)

    @staticmethod
    def from_value(value: Any) -> LiteralValue:
        if isinstance(value, bool):
            return BoolValue(value)
        elif isinstance(value, int):
            return IntValue(value)
        elif isinstance(value, float):
            return FloatValue(value)
        elif isinstance(value, date):
            return DateValue(value)
        elif isinstance(value, datetime):
            return DatetimeValue(value)
        elif isinstance(value, str):
            return StringValue(value)
        elif isinstance(value, list):
            return ArrayValue([LiteralValue.from_value(val) for val in value])
        raise ValueError(f'Unable to find literal value for type {type(value)}')


@dataclass
class IntValue(LiteralValue):
    value: int
    name = 'int'

    @property
    def python_value(self) -> Any:
        return self.value


@dataclass
class FloatValue(LiteralValue):
    value: float
    name = 'float'

    @property
    def python_value(self) -> Any:
        return self.value


@dataclass
class BoolValue(LiteralValue):
    value: bool
    name = 'bool'

    @property
    def python_value(self) -> Any:
        return self.value


@dataclass
class DateValue(LiteralValue):
    value: date
    name = 'date'

    @property
    def python_value(self) -> Any:
        return self.value


@dataclass
class DatetimeValue(LiteralValue):
    value: datetime
    name = 'datetime'

    @property
    def python_value(self) -> Any:
        return self.value


@dataclass
class StringValue(LiteralValue):
    value: str
    name = 'string'

    @property
    def python_value(self) -> Any:
        return self.value


@dataclass
class ArrayValue(LiteralValue):
    value: list[LiteralValue]
    name = 'array'

    @property
    def python_value(self) -> Any:
        return [lit.python_value for lit in self.value]
