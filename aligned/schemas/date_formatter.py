from __future__ import annotations
from dataclasses import dataclass, field
import polars as pl
from polars.type_aliases import TimeUnit
from aligned.schemas.codable import Codable
from mashumaro.types import SerializableType


@dataclass
class AllDateFormatters:

    supported_formatters: dict[str, type[DateFormatter]]

    _shared: AllDateFormatters | None = None

    @classmethod
    def shared(cls) -> AllDateFormatters:
        if cls._shared is None:
            formatters = [
                Timestamp,
                StringDateFormatter,
            ]
            cls._shared = AllDateFormatters({formatter.name(): formatter for formatter in formatters})
        return cls._shared


class DateFormatter(Codable, SerializableType):
    @classmethod
    def name(cls) -> str:
        raise NotImplementedError(cls)

    def decode_polars(self, column: str) -> pl.Expr:
        raise NotImplementedError(type(self))

    def encode_polars(self, column: str) -> pl.Expr:
        raise NotImplementedError(type(self))

    def _serialize(self) -> dict:
        assert type(self).name() in AllDateFormatters.shared().supported_formatters
        data = self.to_dict()
        data['name'] = type(self).name()
        return data

    @classmethod
    def _deserialize(cls, data: dict) -> DateFormatter:
        formatter_name = data.pop('name')
        formatters = AllDateFormatters.shared().supported_formatters
        if formatter_name not in formatters:
            raise ValueError(
                f"Unknown formatter name: {formatter_name}. Supported formatters: {formatters.keys()}"
            )
        formatter_class = formatters[formatter_name]
        return formatter_class.from_dict(data)

    @staticmethod
    def string_format(format: str) -> StringDateFormatter:
        return StringDateFormatter(format)

    @staticmethod
    def iso_8601() -> StringDateFormatter:
        return StringDateFormatter('yyyy-MM-ddTHH:mm:ssZ')

    @staticmethod
    def unix_timestamp(time_unit: TimeUnit = 'us', time_zone: str | None = 'UTC') -> Timestamp:
        return Timestamp(time_unit, time_zone)


@dataclass
class Timestamp(DateFormatter):

    time_unit: TimeUnit = field(default='us')
    time_zone: str | None = field(default='UTC')

    @classmethod
    def name(cls) -> str:
        return 'timestamp'

    def decode_polars(self, column: str) -> pl.Expr:
        if self.time_zone:
            return pl.from_epoch(column, self.time_unit).dt.replace_time_zone(self.time_zone)
        return pl.from_epoch(column, self.time_unit)

    def encode_polars(self, column: str) -> pl.Expr:
        return pl.col(column).dt.timestamp(self.time_unit)


@dataclass
class StringDateFormatter(DateFormatter):

    date_format: str
    time_unit: TimeUnit | None = field(default=None)
    time_zone: str | None = field(default=None)

    @classmethod
    def name(cls) -> str:
        return 'string_form'

    def decode_polars(self, column: str) -> pl.Expr:
        return pl.col(column).str.to_datetime(
            self.date_format, time_unit=self.time_unit, time_zone=self.time_zone
        )

    def encode_polars(self, column: str) -> pl.Expr:
        return pl.col(column).dt.strftime(self.date_format)
