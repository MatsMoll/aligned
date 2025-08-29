from __future__ import annotations

import logging
from typing import Literal, TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, Field

from aligned.schemas.literal_value import LiteralValue
from aligned.schemas.transformation import (
    BinaryOperators,
    BinaryTransformation,
    Expression,
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

    def to_spark_expression(self) -> str:
        spark_op = {
            "NotEq": "!=",
            "Eq": "=",
            "GtEq": ">=",
            "Gt": ">",
            "Lt": "<",
            "LtEq": "<=",
            "Plus": "+",
            "Multiply": "*",
            "TrueDivide": "/",
            "Modulus": "%",
            "Xor": "^",
            "And": "AND",
            "Minus": "-",
            "Or": "OR",
            # "FloorDivide": "/",
        }
        expr = [
            self.left.to_spark_expression(),
            spark_op[self.op],
            self.right.to_spark_expression(),
        ]
        return "(" + " ".join(expr) + ")"

    def to_spark_column(self) -> Column:
        spark_exp = self.to_expression().to_spark()
        assert spark_exp is not None, f"Unable to parse {self} as spark expression"
        return spark_exp


class ScalarValue(BaseModel):
    string: str | None = Field(None, alias="StringOwned")

    def to_spark_expression(self) -> str:
        if self.string:
            return f"'{self.string}'"

        raise ValueError(f"Unable to format '{self}'")

    def to_spark_column(self) -> Column:
        from pyspark.sql.functions import lit

        if self.string:
            return lit(self.string)

        raise ValueError(f"Unable to format '{self}'")

    def to_expression(self) -> Expression:
        return Expression(literal=LiteralValue.from_value(self.string))


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
        if self.string is not None:
            return f"'{self.string}'"
        if self.integer is not None:
            return f"{self.integer}"

        raise ValueError(f"Unable to format '{self}'")

    def to_spark_column(self) -> Column:
        from pyspark.sql.functions import lit

        if self.dynamic:
            return self.dynamic.to_spark_column()
        if self.scalar:
            return self.scalar.value.to_spark_column()
        if self.string is not None:
            return lit(self.string)
        if self.integer is not None:
            return lit(self.integer)

        raise ValueError(f"Unable to format '{self}'")

    def to_expression(self) -> Expression:
        if self.dynamic:
            return self.dynamic.to_expression()
        if self.scalar:
            return self.scalar.value.to_expression()
        if self.string is not None:
            return Expression(literal=LiteralValue.from_value(self.string))
        if self.integer is not None:
            return Expression(literal=LiteralValue.from_value(self.integer))

        raise ValueError(f"Unable to format '{self}'")


class CastExpr(BaseModel):
    expr: ExpressionNode
    dtype: str

    def to_spark_expression(self) -> str:
        expr = self.expr.to_spark_expression()
        return expr

    def to_spark_column(self) -> Column:
        return self.expr.to_spark_column().cast(self.dtype)


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

    def to_spark_column(self) -> Column:
        from pyspark.sql.functions import col

        if self.binary_expr:
            return self.binary_expr.to_spark_column()
        elif self.column:
            return col(self.column)
        elif self.literal:
            return self.literal.to_spark_column()
        elif self.cast:
            return self.cast.to_spark_column()

        raise ValueError(f"Unable to format '{self}'")

    def to_expression(self) -> Expression:
        if self.binary_expr:
            return self.binary_expr.to_expression()
        elif self.column:
            return Expression(column=self.column)
        elif self.literal:
            return self.literal.to_expression()
        elif self.cast:
            raise ValueError(f"Unable to format '{self}'")
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


def polars_expression_to_spark_column(expr: pl.Expr) -> Column | None:
    content = expr.meta.serialize(format="json")
    node = ExpressionNode.model_validate_json(content)
    try:
        return node.to_spark_column()
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
