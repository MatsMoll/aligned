from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

from aligned.schemas.codable import Codable
from mashumaro.types import SerializableType

from typing import TypeVar

T = TypeVar("T")


class ConfigValue(Codable, SerializableType):
    type_name: str

    def read(self) -> str: ...

    def _serialize(self) -> dict:
        assert (
            self.type_name in SupportedValueFactory.shared().supported_values
        ), f"Unknown type_name: {self.type_name}"
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> ConfigValue:
        name_type = value["type_name"]
        if name_type not in SupportedValueFactory.shared().supported_values:
            raise ValueError(
                f"Unknown batch data source id: '{name_type}'.\nRemember to add the"
                " data source to the SupportedValueFactory.supported_values if"
                " it is a custom type."
            )
        del value["type_name"]
        data_class = SupportedValueFactory.shared().supported_values[name_type]
        return data_class.from_dict(value)

    @staticmethod
    def from_value(value: str | ConfigValue) -> ConfigValue:
        if isinstance(value, ConfigValue):
            return value
        else:
            return LiteralValue(value)


class SupportedValueFactory:
    supported_values: dict[str, type[ConfigValue]]

    _shared: SupportedValueFactory | None = None

    def __init__(self) -> None:
        types = [
            EnvironmentValue,
            LiteralValue,
            PlaceholderValue,
            NothingValue,
            ConcatValue,
        ]

        self.supported_values = {val_type.type_name: val_type for val_type in types}

    @classmethod
    def shared(cls) -> SupportedValueFactory:
        if cls._shared:
            return cls._shared
        cls._shared = SupportedValueFactory()
        return cls._shared


@dataclass
class EnvironmentValue(ConfigValue, Codable):
    env: str
    default_value: str | None = field(default=None)
    description: str | None = field(default=None)
    type_name: str = field(default="env")

    def read(self) -> str:
        import os

        if self.default_value is not None and self.env not in os.environ:
            return self.default_value

        try:
            return os.environ[self.env]
        except KeyError as error:
            raise ValueError(f"Missing environment value {self.env}") from error


@dataclass
class LiteralValue(ConfigValue, Codable):
    value: str
    type_name = "literal"

    def read(self) -> str:
        return self.value

    @staticmethod
    def from_value(value: str | ConfigValue) -> ConfigValue:
        if isinstance(value, ConfigValue):
            return value
        else:
            return LiteralValue(value)


@dataclass
class PlaceholderValue(ConfigValue, Codable):
    placeholder_name: str
    type_name = "placeholder"

    def read(self) -> str:
        raise NotImplementedError(
            f"Did not replace placeholder of '{self.placeholder_name}'"
        )


@dataclass
class NothingValue(ConfigValue, Codable):
    type_name = "no"

    def read(self) -> str:
        raise ValueError(
            "Trying to access a config value that is expected to not exist."
        )


@dataclass
class ConcatValue(ConfigValue, Codable):
    values: list[ConfigValue]
    separator = ""
    type_name = "concat"

    def read(self) -> str:
        return self.separator.join([val.read() for val in self.values])


@dataclass
class PathResolver(Codable):
    components: list[ConfigValue]

    @property
    def path(self) -> Path:
        path = Path(self.components[0].read())

        if len(self.components) < 2:  # noqa
            return path

        for comp in self.components[1:]:
            path = path / comp.read()

        return path

    def as_posix(self) -> str:
        return self.path.as_posix()

    def __str__(self) -> str:
        try:
            return self.as_posix()
        except ValueError:
            return " / ".join([str(comp) for comp in self.components])

    def replace(
        self, find: ConfigValue | str, new_value: ConfigValue | str
    ) -> PathResolver:
        find_comp = LiteralValue.from_value(find)
        new_value_comp = LiteralValue.from_value(new_value)
        return PathResolver(
            [new_value_comp if find_comp == comp else comp for comp in self.components]
        )

    def append(self, value: ConfigValue | str) -> PathResolver:
        return PathResolver([*self.components, LiteralValue.from_value(value)])

    @staticmethod
    def from_value(
        values: list[ConfigValue] | str | PathResolver | ConfigValue,
    ) -> PathResolver:
        if isinstance(values, PathResolver):
            return values
        elif isinstance(values, str):
            return PathResolver([LiteralValue.from_value(values)])
        else:
            if isinstance(values, ConfigValue):
                values = [values]
            return PathResolver(values)
