from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import json
from typing import Any, Literal, TYPE_CHECKING, Protocol, Union
from zoneinfo import ZoneInfo

import polars as pl

from aligned.schemas.codable import Codable
from aligned.schemas.constraints import Constraint
from aligned.schemas.literal_value import LiteralValue

if TYPE_CHECKING:
    from aligned.schemas.transformation import Expression
    from aligned.compiler.feature_factory import FeatureFactory
    from pyspark.sql.types import DataType


class StaticFeatureTags:
    is_shadow_model = "is_shadow_model"
    is_model_version = "is_model_version"
    is_entity = "is_entity"

    is_annotated_by = "is_annotated_by"
    is_annotated_feature = "is_annotated_feature"

    is_input_features = "is_input_features"
    "Used to tag which features are used for a log and wait approach"

    is_freshness = "is_updated_at_feature"
    is_image = "is_image_url"
    is_prompt_completion = "is_prompt_completion"

    # Data Types
    is_ordinal = "is_ordinal"
    is_nominal = "is_nomial"
    is_interval = "is_interval"
    is_ratio = "is_ratio"


NAME_POLARS_MAPPING = [
    ("string", pl.Utf8),
    ("int8", pl.Int8),
    ("int16", pl.Int16),
    ("int32", pl.Int32),
    ("int64", pl.Int64),
    ("uint8", pl.UInt8),
    ("uint16", pl.UInt16),
    ("uint32", pl.UInt32),
    ("uint64", pl.UInt64),
    ("float", pl.Float32),
    ("float32", pl.Float32),
    ("float64", pl.Float64),
    ("double", pl.Float64),
    ("bool", pl.Boolean),
    ("date", pl.Date),
    ("datetime", pl.Datetime),
    ("time", pl.Time),
    ("timedelta", pl.Duration),
    ("uuid", pl.Utf8),
    ("array", pl.List(pl.Utf8)),
    ("embedding", pl.List),
    ("json", pl.Utf8),
    ("binary", pl.Binary),
    ("sturct", pl.Struct),
]


@dataclass
class FeatureType(Codable):
    # FIXME: Should use a more Pythonic design, as this one did not behave as intended

    name: str

    @property
    def is_categorical_representable(self) -> bool:
        if "int" in self.name:
            return True
        return self.name in ["string", "bool"]

    @property
    def is_numeric(self) -> bool:
        return self.name in {
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float",
            "float32",
            "float64",
            "double",
        }  # Can be represented as an int

    @property
    def is_datetime(self) -> bool:
        return self.name.startswith("datetime")

    @property
    def is_array(self) -> bool:
        return self.name.startswith("array")

    @property
    def is_struct(self) -> bool:
        return self.name.startswith("struct")

    @property
    def has_structured_fields(self) -> bool:
        return self.name.startswith("struct-")

    def struct_fields(self) -> dict[str, FeatureType]:
        if self.name == "struct":
            return {}

        raw_content = self.name.removeprefix("struct-")
        content = json.loads(raw_content)

        assert isinstance(content, dict)

        return {key: FeatureType(value) for key, value in content.items()}

    def array_subtype(self) -> FeatureType | None:
        if not self.is_array or "-" not in self.name:
            return None

        sub = str(self.name[len("array-") :])
        return FeatureType(sub)

    @property
    def datetime_timezone(self) -> str | None:
        if not self.is_datetime:
            return None

        return self.name.split("-")[1] if "-" in self.name else None

    @property
    def python_type(self) -> type:
        from datetime import date, datetime, time, timedelta
        from uuid import UUID

        from numpy import double

        return {
            "string": str,
            "int8": int,
            "int16": int,
            "int32": int,
            "int64": int,
            "float": float,
            "float32": float,
            "float64": double,
            "double": double,
            "bool": bool,
            "date": date,
            "datetime": datetime,
            "time": time,
            "timedelta": timedelta,
            "uuid": UUID,
            "array": list,
            "embedding": list,
            "json": str,
            "binary": bytes,
        }[self.name]

    @property
    def pandas_type(self) -> str | type:
        import numpy as np

        return {
            "string": str,
            "int8": "Int8",
            "int16": "Int16",
            "int32": "Int32",
            "int64": "Int64",
            "float": np.float32,
            "float32": np.float32,
            "float64": np.float64,
            "double": np.float64,
            "bool": "boolean",
            "date": np.datetime64,
            "datetime": np.datetime64,
            "time": np.datetime64,
            "timedelta": np.timedelta64,
            "uuid": str,
            "array": list,
            "embedding": list,
            "json": str,
            "binary": bytes,
        }[self.name]

    @property
    def spark_type(self) -> DataType:
        from pyspark.sql.types import (
            ArrayType,
            BooleanType,
            ByteType,
            DoubleType,
            FloatType,
            IntegerType,
            LongType,
            ShortType,
            StringType,
            DateType,
            StructField,
            StructType,
            TimestampType,
            BinaryType,
            MapType,
        )

        if self.is_datetime:
            return TimestampType()

        if self.is_struct:
            if not self.has_structured_fields:
                return MapType(StringType(), StringType())
            else:
                sub_fields = self.struct_fields()
                return StructType(
                    [
                        StructField(name=key, dataType=dtype.spark_type)
                        for key, dtype in sorted(
                            sub_fields.items(), key=lambda vals: vals[0]
                        )
                    ]
                )

        if self.is_array:
            sub_type = self.array_subtype()
            if sub_type:
                return ArrayType(sub_type.spark_type)
            else:
                return ArrayType(StringType())

        if self.is_embedding:
            return ArrayType(FloatType())

        spark_dtypes = [
            ("string", StringType()),
            ("int8", ByteType()),
            ("int16", ShortType()),
            ("int32", IntegerType()),
            ("int64", LongType()),
            ("uint8", ByteType()),
            ("uint16", ShortType()),
            ("uint32", IntegerType()),
            ("uint64", LongType()),
            ("float", FloatType()),
            ("float32", FloatType()),
            ("float64", DoubleType()),
            ("double", DoubleType()),
            ("bool", BooleanType()),
            ("date", DateType()),
            ("datetime", TimestampType()),
            ("uuid", StringType()),
            ("json", StringType()),
            ("binary", BinaryType()),
        ]
        for name, dtype in spark_dtypes:
            if name == self.name:
                return dtype

        raise ValueError(f"Unable to find a value that can represent {self.name}")

    @property
    def polars_type(self) -> pl.DataType:
        if self.is_datetime:
            time_zone = self.datetime_timezone
            return pl.Datetime(time_zone=time_zone)  # type: ignore

        if self.is_struct:
            if not self.has_structured_fields:
                return pl.Struct([])
            else:
                return pl.Struct(
                    [
                        pl.Field(name=name, dtype=dtype.polars_type)
                        for name, dtype in self.struct_fields().items()
                    ]
                )

        if self.is_array:
            sub_type = self.array_subtype()
            if sub_type:
                return pl.List(sub_type.polars_type)  # type: ignore
            else:
                return pl.List(pl.Utf8)  # type: ignore

        if self.is_embedding:
            return pl.List(pl.Float64)  # type: ignore

        for name, dtype in NAME_POLARS_MAPPING:
            if name == self.name:
                return dtype

        raise ValueError(f"Unable to find a value that can represent {self.name}")

    @property
    def feature_factory(self) -> FeatureFactory:
        from aligned.compiler import feature_factory as ff

        if self.name.startswith("datetime-"):
            time_zone = self.name.split("-")[1]
            return ff.Timestamp(time_zone=time_zone)

        if self.name.startswith("array-"):
            sub_type = "-".join(self.name.split("-")[1:])
            return ff.List(FeatureType(name=sub_type).feature_factory)

        if self.is_embedding:
            embedding_size = self.embedding_size()
            if embedding_size:
                return ff.Embedding(embedding_size=embedding_size)

        return {
            "string": ff.String(),
            "int8": ff.Int8(),
            "int16": ff.Int16(),
            "int32": ff.Int32(),
            "int64": ff.Int64(),
            "uint8": ff.UInt8(),
            "uint16": ff.UInt16(),
            "uint32": ff.UInt32(),
            "uint64": ff.UInt64(),
            "float": ff.Float32(),
            "float32": ff.Float32(),
            "float64": ff.Float64(),
            "double": ff.Float64(),
            "bool": ff.Bool(),
            "date": ff.Timestamp(),
            "datetime": ff.Timestamp(),
            "time": ff.Timestamp(),
            "timedelta": ff.Timestamp(),
            "uuid": ff.UUID(),
            "array": ff.List(ff.String()),
            "embedding": ff.Embedding(768),
            "binary": ff.Binary(),
            "json": ff.Json(),
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
        if isinstance(polars_type, pl.Null) or isinstance(polars_type, pl.Unknown):
            return FeatureType.string()

        if isinstance(polars_type, pl.Datetime):
            if polars_type.time_zone:
                return FeatureType(name=f"datetime-{polars_type.time_zone}")
            return FeatureType(name="datetime")

        if isinstance(polars_type, pl.List):
            if polars_type.inner:
                sub_type = FeatureType.from_polars(polars_type.inner)  # type: ignore
                return FeatureType(name=f"array-{sub_type.name}")

            return FeatureType(name="array")

        if isinstance(polars_type, pl.Struct):
            return FeatureType.struct(
                {
                    field.name: FeatureType.from_polars(field.dtype)  # type: ignore
                    for field in polars_type.fields
                }
            )

        for name, dtype in NAME_POLARS_MAPPING:
            if polars_type.is_(dtype):
                return FeatureType(name=name)

        raise ValueError(f"Unable to find a value that can represent {polars_type}")

    @staticmethod
    def from_type(dtype: Any) -> FeatureType | None:
        from pydantic import BaseModel
        from pydantic.fields import FieldInfo
        from aligned.feature_view.feature_view import FeatureViewWrapper
        from dataclasses import fields, is_dataclass
        from typing import get_origin, get_args
        from inspect import getmro

        if dtype in [str, "str"]:
            return FeatureType.string()
        elif dtype in [int, "int"]:
            return FeatureType.int64()
        elif dtype in [float, "float"]:
            return FeatureType.floating_point()
        elif dtype in [date, "date"]:
            return FeatureType.date()
        elif dtype in [datetime, "datetime"]:
            return FeatureType.datetime()
        elif dtype in [bool, "bool"]:
            return FeatureType.boolean()
        elif get_origin(dtype) is list:
            args = get_args(dtype)
            if args:
                return FeatureType.array(FeatureType.from_type(args[0]))
            else:
                return FeatureType.array()

        if isinstance(dtype, list) and dtype and isinstance(dtype[0], Feature):
            all_fields = {}
            for field in dtype:
                all_fields[field.name] = field.dtype
            return FeatureType.struct(all_fields)

        # Needs to be before is_dataclass
        if isinstance(dtype, FeatureViewWrapper):
            all_fields = {}
            for field in dtype.query().request.all_returned_features:
                all_fields[field.name] = field.dtype
            return FeatureType.struct(all_fields)

        if is_dataclass(dtype):
            all_fields = {}
            for field in fields(dtype):
                all_fields[field.name] = FeatureType.from_type(field.type)

            return FeatureType.struct(all_fields)

        if BaseModel in getmro(dtype):  # type: ignore
            all_fields = {}
            model_fields: dict[str, FieldInfo] = dtype.model_fields  # type: ignore
            assert isinstance(model_fields, dict)
            for name, field in model_fields.items():
                all_fields[name] = FeatureType.from_type(field.annotation)
            return FeatureType.struct(all_fields)

        return None

    @staticmethod
    def binary() -> FeatureType:
        return FeatureType(name="binary")

    @staticmethod
    def string() -> FeatureType:
        return FeatureType(name="string")

    @staticmethod
    def uint8() -> FeatureType:
        return FeatureType(name="uint8")

    @staticmethod
    def uint16() -> FeatureType:
        return FeatureType(name="uint16")

    @staticmethod
    def uint32() -> FeatureType:
        return FeatureType(name="uint32")

    @staticmethod
    def uint64() -> FeatureType:
        return FeatureType(name="uint64")

    @staticmethod
    def int8() -> FeatureType:
        return FeatureType(name="int8")

    @staticmethod
    def int16() -> FeatureType:
        return FeatureType(name="int16")

    @staticmethod
    def int32() -> FeatureType:
        return FeatureType(name="int32")

    @staticmethod
    def boolean() -> FeatureType:
        return FeatureType(name="bool")

    @staticmethod
    def int64() -> FeatureType:
        return FeatureType(name="int64")

    @staticmethod
    def float64() -> FeatureType:
        return FeatureType(name="float64")

    @staticmethod
    def float32() -> FeatureType:
        return FeatureType(name="float32")

    @staticmethod
    def floating_point() -> FeatureType:
        return FeatureType(name="float")

    @staticmethod
    def double() -> FeatureType:
        return FeatureType(name="double")

    @staticmethod
    def date() -> FeatureType:
        return FeatureType(name="date")

    @staticmethod
    def uuid() -> FeatureType:
        return FeatureType(name="uuid")

    @staticmethod
    def datetime(tz: ZoneInfo | None = ZoneInfo("UTC")) -> FeatureType:
        if not tz:
            return FeatureType(name="datetime")
        return FeatureType(name=f"datetime-{tz.key}")

    @staticmethod
    def json() -> FeatureType:
        return FeatureType(name="json")

    @staticmethod
    def struct(subtypes: dict[str, FeatureType] | None = None) -> FeatureType:
        if subtypes is None:
            return FeatureType(name="struct")

        content = json.dumps({key: value.name for key, value in subtypes.items()})
        return FeatureType(name=f"struct-{content}")

    @staticmethod
    def array(sub_type: FeatureType | None = None) -> FeatureType:
        if sub_type is None:
            return FeatureType(name="array")
        return FeatureType(name=f"array-{sub_type.name}")

    @staticmethod
    def embedding(size: int) -> FeatureType:
        return FeatureType(name=f"embedding-{size}")

    @property
    def is_embedding(self) -> bool:
        return self.name.startswith("embedding")

    def embedding_size(self) -> int | None:
        if "-" not in self.name:
            return None
        return int(self.name.split("-")[1])


@dataclass
class Feature(Codable):
    name: str
    dtype: FeatureType
    description: str | None = None
    tags: list[str] | None = None

    constraints: set[Constraint] | None = None
    default_value: LiteralValue | None = None

    def __pre_serialize__(self) -> Feature:
        assert isinstance(self.name, str)
        assert isinstance(self.dtype, FeatureType)
        assert isinstance(self.description, str) or self.description is None
        assert isinstance(self.tags, list) or self.tags is None
        assert (
            isinstance(self.default_value, LiteralValue) or self.default_value is None
        )
        if self.constraints:
            for constraint in self.constraints:
                assert isinstance(constraint, Constraint)

        return self

    def to_expression(self) -> Expression:
        from aligned.schemas.transformation import Expression

        return Expression(column=self.name)

    def renamed(self, new_name: str) -> Feature:
        return Feature(
            name=new_name,
            dtype=self.dtype,
            description=self.description,
            tags=self.tags,
            constraints=self.constraints,
            default_value=self.default_value,
        )

    def as_reference(self, location: FeatureLocation) -> FeatureReference:
        return FeatureReference(name=self.name, location=location)

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        value = f"{self.name} - {self.dtype.name}"
        if self.description:
            value += f" - {self.description}"
        return value


@dataclass
class EventTimestamp(Codable):
    name: str
    ttl: int | None = None
    description: str | None = None
    tags: set[str] | None = None
    dtype: FeatureType = field(default_factory=lambda: FeatureType.datetime())

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        value = f"{self.name} - {self.dtype.name}"
        if self.description:
            value += f" - {self.description}"
        return value

    def as_feature(self) -> Feature:
        return Feature(
            name=self.name,
            dtype=self.dtype,
            description=self.description,
            tags=list(self.tags or set()),
        )


@dataclass
class FeatureLocation(Codable):
    name: str
    location_type: Literal["feature_view", "combined_view", "model"]

    @property
    def identifier(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.location_type}:{self.name}"

    def __hash__(self) -> int:
        return (self.name + self.location_type).__hash__()

    @staticmethod
    def feature_view(name: str) -> FeatureLocation:
        return FeatureLocation(name, "feature_view")

    @staticmethod
    def combined_view(name: str) -> FeatureLocation:
        return FeatureLocation(name, "combined_view")

    @staticmethod
    def model(name: str) -> FeatureLocation:
        return FeatureLocation(name, "model")

    @staticmethod
    def from_string(string: str) -> FeatureLocation:
        splits = string.split(":")
        location_type = splits[0]
        assert location_type in [
            "feature_view",
            "combined_view",
            "model",
        ], f"Unexpected location type {location_type}"
        return FeatureLocation(name=splits[1], location_type=location_type)  # type: ignore


class LocationReferencable(Protocol):
    @property
    def location(self) -> FeatureLocation: ...


ConvertableToLocation = Union[FeatureLocation, str, LocationReferencable]


def convert_to_location(location: ConvertableToLocation) -> FeatureLocation:
    if isinstance(location, FeatureLocation):
        return location

    if isinstance(location, str):
        return FeatureLocation.from_string(location)

    return location.location


class FeatureReferencable:
    def feature_reference(self) -> FeatureReference:
        raise NotImplementedError(type(self))


@dataclass
class FeatureReference(Codable, FeatureReferencable):
    name: str
    location: FeatureLocation

    def as_feature(self) -> Feature:
        return Feature(
            name=self.name,
            dtype=FeatureType.string(),
            description=None,
            tags=None,
            constraints=None,
        )

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def identifier(self) -> str:
        return f"{self.location.identifier}:{self.name}"

    def feature_reference(self) -> FeatureReference:
        return self

    @staticmethod
    def from_string(value: str) -> FeatureReference | None:
        if ":" not in value:
            return None

        splits = value.split(":")
        if len(splits) != 3:
            return None

        loc = FeatureLocation.from_string(":".join(splits[0:2]))
        if loc is None:
            return None

        return FeatureReference(name=splits[-1], location=loc)
