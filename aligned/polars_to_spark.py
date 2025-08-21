from __future__ import annotations

import logging
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


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
            "Eq": "=",
            "GtEq": ">=",
            "Gt": ">",
            "Lt": "<",
            "LtEq": "<=",
            "Plus": "+",
            "Multiply": "*",
            "TrueDivide": "/",
            "Modulus": "/",
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
