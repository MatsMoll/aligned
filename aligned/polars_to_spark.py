from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import polars as pl
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pyspark.sql.types import DataType, StructType


logger = logging.getLogger(__name__)


def polars_schema_to_spark(schema: dict[str, pl.PolarsDataType]) -> StructType:
    from pyspark.sql.types import StructField, StructType

    return StructType(
        [
            StructField(name=name, dataType=polars_dtype_to_spark(dtype))
            for name, dtype in schema.items()
        ]
    )


def polars_dtype_to_spark(data_type: pl.PolarsDataType) -> DataType:  # noqa: PLR0911
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
        StructField,
        StructType,
        TimestampType,
    )

    if isinstance(data_type, pl.String):
        return StringType()
    if isinstance(data_type, pl.Float32):
        return FloatType()
    if isinstance(data_type, pl.Float64):
        return DoubleType()
    if isinstance(data_type, pl.Int8):
        return ByteType()
    if isinstance(data_type, pl.Int16):
        return ShortType()
    if isinstance(data_type, pl.Int32):
        return IntegerType()
    if isinstance(data_type, pl.Int64):
        return LongType()
    if isinstance(data_type, pl.Boolean):
        return BooleanType()
    if isinstance(data_type, pl.Datetime):
        return TimestampType()
    if isinstance(data_type, (pl.Array, pl.List)):
        if data_type.inner:
            return ArrayType(polars_dtype_to_spark(data_type.inner))
        return ArrayType(StringType())
    if isinstance(data_type, pl.Struct):
        return StructType(
            [
                StructField(
                    name=field.name, dataType=polars_dtype_to_spark(field.dtype)
                )
                for field in data_type.fields
            ]
        )

    raise ValueError(f"Unsupported type {data_type}")


class BinaryExpression(BaseModel):
    left: ExpressionNode
    op: Literal[
        "NotEq",
        "Eq",
        "GtEq",
        "Gt",
        "Lt",
        "LtEq",
        "Plus",
        "Multiply",
        "TrueDivide",
        "FloorDivide",
        "Modulus",
        "Xor",
        "And",
        "Or",
        "Minus",
    ]
    right: ExpressionNode

    def to_spark_expression(self) -> str:
        spark_op = {
            "NotEq": "!=",
            "Eq": "==",
            "GtEq": ">=",
            "Gt": ">",
            "Lt": "<",
            "LtEq": "<=",
            "Plus": "+",
            "Multiply": "*",
            "TrueDivide": "/",
            "Modulus": "/",
            "Xor": "^",
            "And": "&",
            "Minus": "-",
            # "Or": "|",
            # "FloorDivide": "/",
        }
        expr = [
            self.left.to_spark_expression(),
            spark_op[self.op],
            self.right.to_spark_expression(),
        ]
        return " ".join(expr)


class ScalarValue(BaseModel):
    string: str | None = Field(None, alias="StringOwned")

    def to_spark_expression(self) -> str:
        if self.string:
            return f"'{self.string}'"

        raise ValueError(f"Unable to format '{self}'")


class Scalar(BaseModel):
    value: ScalarValue


class LiteralPolarsValue(BaseModel):
    string: str | None = Field(None, alias="String")
    integer: int | None = Field(None, alias="Int")
    dynamic: LiteralPolarsValue | None = Field(None, alias="Dyn")
    scalar: Scalar | None = Field(None, alias="Scalar")

    def to_spark_expression(self) -> str:
        if self.dynamic:
            return self.dynamic.to_spark_expression()
        if self.scalar:
            return self.scalar.value.to_spark_expression()
        if self.string:
            return f"'{self.string}'"
        if self.integer:
            return f"{self.integer}"

        raise ValueError(f"Unable to format '{self}'")


class CastExpr(BaseModel):
    expr: ExpressionNode
    dtype: str

    def to_spark_expression(self) -> str:
        expr = self.expr.to_spark_expression()
        return expr


class ExpressionNode(BaseModel):
    binary_expr: BinaryExpression | None = Field(None, alias="BinaryExpr")
    column: str | None = Field(None, alias="Column")
    literal: LiteralPolarsValue | None = Field(None, alias="Literal")
    cast: CastExpr | None = Field(None, alias="Cast")

    def to_spark_expression(self) -> str:
        if self.binary_expr:
            return self.binary_expr.to_spark_expression()
        elif self.column:
            return self.column
        elif self.literal:
            return self.literal.to_spark_expression()
        elif self.cast:
            return self.cast.to_spark_expression()

        raise ValueError(f"Unable to format '{self}'")


def polars_expression_to_spark(expr: pl.Expr) -> str | None:
    content = expr.meta.serialize(format="json")
    node = ExpressionNode.model_validate_json(content)
    try:
        return node.to_spark_expression()
    except ValueError:
        logger.error(
            f"Unable to transform expression {content}, which was decoded {node}"
        )
        return None
