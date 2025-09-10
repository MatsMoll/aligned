from __future__ import annotations

import logging
from typing import Any, Literal, TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, Field

from aligned.schemas.feature import FeatureType
from aligned.schemas.literal_value import LiteralValue
from aligned.schemas.transformation import (
    BinaryOperators,
    BinaryTransformation,
    Expression,
    Transformation,
    UnaryFunction,
    UnaryTransformation,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyspark.sql.functions import Column


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

    def to_expression(self) -> Expression:
        op_mapping: dict[str, BinaryOperators] = {
            "NotEq": "neq",
            "Eq": "eq",
            "GtEq": "gte",
            "Gt": "gt",
            "Lt": "lt",
            "LtEq": "lte",
            "Plus": "add",
            "Multiply": "mul",
            "TrueDivide": "div",
            "Modulus": "mod",
            "Xor": "xor",
            "And": "and",
            "Minus": "sub",
            "Or": "or",
            "Pow": "pow",
        }
        return Expression(
            transformation=BinaryTransformation(
                left=self.left.to_expression(),
                right=self.right.to_expression(),
                operator=op_mapping[self.op],
            )
        )


class Scalar(BaseModel):
    value: LiteralPolarsValue | None = Field(None, alias="value")

    def to_expression(self) -> Expression:
        if self.value is not None:
            return self.value.to_expression()
        raise ValueError(f"Unable to format '{self}'")


class LiteralPolarsValue(BaseModel):
    string_owned: str | None = Field(None, alias="StringOwned")
    string: str | None = Field(None, alias="String")
    integer: int | None = Field(None, alias="Int")
    dynamic: LiteralPolarsValue | None = Field(None, alias="Dyn")
    floating_point: float | None = Field(None, alias="Float")
    scalar: LiteralPolarsValue | None = Field(None, alias="Scalar")
    boolean: bool | None = Field(None, alias="Boolean")
    value: LiteralPolarsValue | None = Field(None)

    def to_expression(self) -> Expression:
        if self.dynamic:
            return self.dynamic.to_expression()
        if self.scalar:
            return self.scalar.to_expression()
        if self.boolean is not None:
            return Expression(literal=LiteralValue.from_value(self.boolean))
        if self.string is not None:
            return Expression(literal=LiteralValue.from_value(self.string))
        if self.string_owned is not None:
            return Expression(literal=LiteralValue.from_value(self.string_owned))
        if self.integer is not None:
            return Expression(literal=LiteralValue.from_value(self.integer))
        if self.floating_point is not None:
            return Expression(literal=LiteralValue.from_value(self.floating_point))
        if self.value is not None:
            return self.value.to_expression()

        raise ValueError(f"Unable to format '{self}'")


class Dtype(BaseModel):
    literal: str = Field(alias="Literal")


class CastExpr(BaseModel):
    expr: ExpressionNode
    dtype: str | Dtype

    def to_expression(self) -> Expression:
        from aligned.schemas.feature import NAME_POLARS_MAPPING
        from aligned.schemas.transformation import CastTransform

        dtype_name = self.dtype if isinstance(self.dtype, str) else self.dtype.literal

        for name, dtype in NAME_POLARS_MAPPING:
            if dtype_name.lower() == name:
                return Expression(
                    transformation=CastTransform(
                        inner=self.expr.to_expression(),
                        dtype=FeatureType.from_polars(dtype),
                    )
                )
        raise ValueError(f"Unable to cast {self}")


class BooleanFunctions(BaseModel):
    is_between: dict[str, Any] | None = Field(None, alias="IsBetween")

    def to_expression(self, input: list[ExpressionNode]) -> Expression:
        from aligned.schemas.transformation import IsBetweenTransformation

        tran: Transformation | None = None
        if self.is_between:
            tran = IsBetweenTransformation(
                value=input[0].to_expression(),
                lower_bound=input[1].to_expression(),
                upper_bound=input[2].to_expression(),
            )

        assert tran is not None
        return Expression(transformation=tran)


class LogOptions(BaseModel):
    base: float

    def to_expression(self, input: list[ExpressionNode]) -> Expression:
        from aligned.schemas.transformation import Log

        return Expression(transformation=Log(input[0].to_expression(), self.base))


class StringFunctions(BaseModel):
    contains: dict[str, Any] | None = Field(None, alias="Contains")

    def to_expression(self, input: list[ExpressionNode]) -> Expression:
        tran: Transformation | None = None
        if self.contains is not None:
            tran = BinaryTransformation(
                left=input[0].to_expression(),
                right=input[1].to_expression(),
                operator="str_contains",
            )

        assert tran
        return Expression(transformation=tran)


BoolFunctions = Literal["IsNan", "Not", "IsNull", "IsNotNull", "IsNotNan"]
StrFunctions = Literal[
    "StartsWith", "EndsWith", "Contains", "LenChars", "Lowercase", "Uppercase"
]
UnaryFunctions = Literal["Exp", "Abs", "Log"]
TrigFunctions = Literal["Sin", "Cos", "Tan"]
TemporalFunctions = Literal[
    "Day",
    "Days",
    "Hour",
    "Hours",
    "Minute",
    "Minutes",
    "Second",
    "Seconds",
    "Week",
    "Weekday",
    "Year",
    "DayOfWeek",
    "IsoYear",
]


class FunctionOptions(BaseModel):
    boolean: BooleanFunctions | BoolFunctions | None = Field(None, alias="Boolean")
    string: StringFunctions | StrFunctions | None = Field(None, alias="StringExpr")
    log: LogOptions | None = Field(None, alias="Log")
    pow: str | None = Field(None, alias="Pow")
    trig: TrigFunctions | None = Field(None, alias="Trigonometry")
    temporal: str | None = Field(None, alias="TemporalExpr")


class FunctionNode(BaseModel):
    input: list[ExpressionNode]
    function: FunctionOptions | UnaryFunctions

    def to_expression(self) -> Expression:
        tran: Transformation | None = None
        if isinstance(self.function, str):
            func_mapping: dict[UnaryFunctions, UnaryFunction] = {
                "Exp": "exp",
                "Abs": "abs",
                "Log": "log",
            }
            tran = UnaryTransformation(
                inner=self.input[0].to_expression(), func=func_mapping[self.function]
            )
        elif self.function.log:
            return self.function.log.to_expression(self.input)
        elif self.function.pow is not None:
            match self.function.pow:
                case "Generic":
                    tran = BinaryTransformation(
                        left=self.input[0].to_expression(),
                        right=self.input[1].to_expression(),
                        operator="pow",
                    )
                case "Sqrt":
                    tran = UnaryTransformation(
                        inner=self.input[0].to_expression(), func="sqrt"
                    )
                case _:
                    raise NotImplementedError(self)
        elif self.function.boolean:
            if isinstance(self.function.boolean, str):
                mapping: dict[BoolFunctions, UnaryFunction] = {
                    "Not": "not",
                    "IsNan": "is_nan",
                    "IsNotNan": "is_not_nan",
                    "IsNull": "is_null",
                    "IsNotNull": "is_not_null",
                }
                tran = UnaryTransformation(
                    inner=self.input[0].to_expression(),
                    func=mapping[self.function.boolean],
                )
            else:
                return self.function.boolean.to_expression(self.input)
        elif self.function.string:
            if isinstance(self.function.string, str):
                bin_str_mapping: dict[StrFunctions, BinaryOperators] = {
                    "StartsWith": "str_starts_with",
                    "EndsWith": "str_ends_with",
                    "Contains": "str_contains",
                }
                un_str_mapping: dict[StrFunctions, UnaryFunction] = {
                    "LenChars": "str_len_char",
                    "Lowercase": "str_to_lower",
                    "Uppercase": "str_to_upper",
                }

                if self.function.string in bin_str_mapping:
                    tran = BinaryTransformation(
                        self.input[0].to_expression(),
                        self.input[1].to_expression(),
                        operator=bin_str_mapping[self.function.string],
                    )
                else:
                    tran = UnaryTransformation(
                        self.input[0].to_expression(),
                        func=un_str_mapping[self.function.string],
                    )
            else:
                return self.function.string.to_expression(self.input)
        elif self.function.trig:
            if not isinstance(self.function.trig, str):
                raise NotImplementedError(self)

            trig_mapping: dict[TrigFunctions, UnaryFunction] = {
                "Sin": "sin",
                "Cos": "cos",
                "Tan": "tan",
            }

            tran = UnaryTransformation(
                self.input[0].to_expression(), func=trig_mapping[self.function.trig]
            )
        elif self.function.temporal:
            from aligned.schemas.transformation import DateComponent

            exp = self.input[0].to_expression()
            assert (
                exp.column
            ), "Currently only supporting column transformation on DateComponent"

            tran = DateComponent(exp.column, self.function.temporal.lower())

        assert tran
        return Expression(transformation=tran)


class ExpressionNode(BaseModel):
    binary_expr: BinaryExpression | None = Field(None, alias="BinaryExpr")
    column: str | None = Field(None, alias="Column")
    literal: LiteralPolarsValue | None = Field(None, alias="Literal")
    cast: CastExpr | None = Field(None, alias="Cast")
    function: FunctionNode | None = Field(None, alias="Function")

    def to_expression(self) -> Expression:
        if self.binary_expr:
            return self.binary_expr.to_expression()
        elif self.column:
            return Expression(column=self.column)
        elif self.literal:
            return self.literal.to_expression()
        elif self.cast:
            return self.cast.to_expression()
        elif self.function:
            return self.function.to_expression()

        raise ValueError(f"Unable to format '{self}'")


def polars_expression_to_spark_column(expr: pl.Expr) -> Column | None:
    content = expr.meta.serialize(format="json")
    node = ExpressionNode.model_validate_json(content)
    try:
        return node.to_expression().to_spark()
    except ValueError:
        logger.error(
            f"Unable to transform expression {content}, which was decoded {node}"
        )
        return None


def polars_to_expression(expr: pl.Expr) -> Expression | None:
    content = expr.meta.serialize(format="json")
    node = ExpressionNode.model_validate_json(content)
    try:
        return node.to_expression()
    except ValueError as e:
        logger.exception(e)
        logger.error(
            f"Unable to transform expression {content}, which was decoded {node}"
        )
        return None
