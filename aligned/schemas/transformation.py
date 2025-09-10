from __future__ import annotations

import logging
import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import polars as pl
from mashumaro.types import SerializableType
from sqlglot import exp

from aligned.lazy_imports import pandas as pd
from aligned.schemas.codable import Codable
from aligned.schemas.feature import FeatureReference, FeatureType
from aligned.schemas.literal_value import LiteralValue
from aligned.schemas.text_vectoriser import EmbeddingModel

if TYPE_CHECKING:
    from aligned.sources.s3 import AwsS3Config
    from aligned.feature_store import ContractStore

    from pyspark.sql import Column, DataFrame as SparkFrame


logger = logging.getLogger(__name__)


@dataclass
class TransformationTestDefinition:
    transformation: Transformation
    input: dict[str, list]
    output: list

    @property
    def input_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.input)

    @property
    def output_pandas(self) -> pd.Series:
        return pd.Series(self.output)

    @property
    def input_polars(self) -> pl.DataFrame:
        return pl.from_dict(self.input, strict=False)

    @property
    def output_polars(self) -> pl.Series:
        try:
            values = pl.Series(
                self.output, dtype=self.transformation.dtype.polars_type
            ).fill_nan(None)
            if self.transformation.dtype == FeatureType.boolean():
                return values.cast(pl.Boolean)
            else:
                return values
        except pl.exceptions.InvalidOperationError:
            return pl.Series(self.output, dtype=self.transformation.dtype.polars_type)


def gracefull_transformation(
    df: pd.DataFrame,
    is_valid_mask: pd.Series,
    transformation: Callable[[pd.Series], pd.Series],
) -> pd.Series:
    result = pd.Series(np.repeat(np.nan, repeats=is_valid_mask.shape[0]))
    return result.mask(is_valid_mask, transformation(df.loc[is_valid_mask]))


class PsqlTransformation:
    def as_psql(self) -> str:
        raise NotImplementedError()


class RedshiftTransformation:
    def as_redshift(self) -> str:
        if isinstance(self, PsqlTransformation):
            return self.as_psql()
        raise NotImplementedError()


class GlotExprTransformation:
    def to_glot(self) -> exp.Expression:
        raise NotImplementedError(type(self))


class PolarsExprTransformation:
    def polars_expr(self) -> pl.Expr | None:
        raise NotImplementedError(type(self))


class SparkExpression:
    def spark_col(self) -> Column | None:
        raise NotImplementedError(type(self))


class InnerTransformation(PolarsExprTransformation, SparkExpression):
    """
    A general representation of transformations that transforms one value.

    E.g. Is null, taking the absolute or rounding
    """

    inner: Expression

    def polars_expr_from(self, inner: pl.Expr) -> pl.Expr:
        raise NotImplementedError(type(self))

    def polars_expr(self) -> pl.Expr | None:
        inner_exp = self.inner.to_polars()
        if inner_exp is not None:
            return self.polars_expr_from(inner_exp)
        else:
            return None

    def pandas_tran(self, column: pd.Series) -> pd.Series:
        raise NotImplementedError(type(self))

    def spark_col_from(self, inner: Column) -> Column | None:
        raise NotImplementedError(type(self))

    def spark_col(self) -> Column | None:
        inner_exp = self.inner.to_spark()
        if inner_exp is not None:
            return self.spark_col_from(inner_exp)
        else:
            return None


class Transformation(Codable, SerializableType):
    name: str
    dtype: FeatureType

    def needed_columns(self) -> list[str]:
        raise NotImplementedError(self)

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        if isinstance(self, InnerTransformation):
            if self.inner.column:
                return self.pandas_tran(df[self.inner.column])  # type: ignore
            if self.inner.transformation:
                inner = await self.inner.transformation.transform_pandas(df, store)
                return self.pandas_tran(inner)

            raise ValueError(
                f"Unable to transform literal value with inner transformation. {type(self)}. "
                "Consider precomputing the value."
            )

        raise NotImplementedError(type(self))

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        if isinstance(self, PolarsExprTransformation):
            exp = self.polars_expr()
            if exp is not None:
                return exp

        if isinstance(self, InnerTransformation):
            assert self.inner.transformation is not None
            output_key = "_aligned_out"
            inner = await self.inner.transformation.transform_polars(
                df, output_key, store
            )
            if isinstance(inner, pl.Expr):
                return self.polars_expr_from(inner)
            else:
                return df.with_columns(
                    self.polars_expr_from(pl.col(output_key))
                ).select(pl.exclude(output_key))

        raise NotImplementedError(type(self))

    async def transform_spark(
        self, df: SparkFrame, alias: str, store: ContractStore
    ) -> SparkFrame:
        if isinstance(self, SparkExpression):
            exp = self.spark_col()
            if exp is not None:
                return df.withColumn(alias, exp)

        raise NotImplementedError(type(self))

    def _serialize(self) -> dict:
        return self.to_dict()

    def should_skip(self, output_column: str, columns: list[str]) -> bool:
        return output_column in columns

    @classmethod
    def _deserialize(cls, value: dict) -> Transformation:
        name_type = value["name"]
        del value["name"]
        data_class = SupportedTransformations.shared().types[name_type]
        with suppress(AttributeError):
            if data_class.dtype:
                del value["dtype"]

        return data_class.from_dict(value)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        raise NotImplementedError()

    @classmethod
    async def run_transformation_test_polars(cls) -> None:
        from polars.testing import assert_series_equal
        from aligned import ContractStore

        try:
            test = cls.test_definition()
            alias = "something"
            output_df = await test.transformation.transform_polars(
                test.input_polars.lazy(), alias=alias, store=ContractStore.empty()
            )
            if isinstance(output_df, pl.Expr):
                output_df = test.input_polars.lazy().with_columns(
                    [output_df.alias(alias)]
                )
            output = output_df.select(pl.col(alias)).collect().to_series()

            missing_columns = set(test.input_polars.columns) - set(
                output_df.collect_schema().names()
            )
            assert missing_columns == set(), f"Missing columns: {missing_columns}"

            expected = test.output_polars
            if test.transformation.dtype == FeatureType.boolean():
                is_correct = output.equals(test.output_polars.alias(alias))
                assert is_correct, (
                    f"Output for {cls.__name__} is not correct.,"
                    f"\nGot: {output},\nexpected: {test.output_polars}"
                )
            else:
                assert_series_equal(
                    expected.alias(alias), output, check_names=False, check_dtypes=False
                )
        except AttributeError:
            raise AssertionError(
                f"Error for transformation {cls.__name__}. Could be missing a return in the transformation"
            )
        except NotImplementedError:
            pass
        except TypeError as e:
            raise ValueError(f"Error for transformation {cls.__name__}: {e}")

    @classmethod
    async def run_transformation_test_pandas(cls) -> None:
        import numpy as np
        from aligned import ContractStore
        from numpy.testing import assert_almost_equal

        with suppress(NotImplementedError):
            test = cls.test_definition()
            output = await test.transformation.transform_pandas(
                test.input_pandas, ContractStore.empty()
            )
            if test.transformation.dtype == FeatureType.boolean():
                is_correct = np.all(output == test.output_pandas) | output.equals(
                    test.output_pandas
                )
                assert is_correct, (
                    f"Output for {cls.__name__} is not correct.,"
                    f"\nGot: {output},\nexpected: {test.output_pandas}"
                )
            elif test.transformation.dtype == FeatureType.string():
                expected = test.output_pandas
                assert expected.equals(output), (
                    f"Output for {cls.__name__} is not correct.,"
                    f"\nGot: {output},\nexpected: {test.output_pandas}"
                )
            else:
                expected = test.output_pandas.to_numpy()
                output_np = output.to_numpy().astype("float")
                is_null = np.isnan(expected) & np.isnan(output_np)
                assert_almost_equal(expected[~is_null], output_np[~is_null])


class SupportedTransformations:
    types: dict[str, type[Transformation]]

    _shared: SupportedTransformations | None = None

    def __init__(self) -> None:
        self.types = {}

        for tran_type in [
            PandasLambdaTransformation,
            PandasFunctionTransformation,
            PolarsLambdaTransformation,
            PolarsExpression,
            StructField,
            Contains,
            DateComponent,
            TimeDifference,
            ToNumerical,
            HashColumns,
            ReplaceStrings,
            MultiTransformation,
            IsIn,
            BinaryTransformation,
            UnaryTransformation,
            Ordinal,
            FillNaValues,
            FillNaValuesColumns,
            CopyTransformation,
            WordVectoriser,
            MapArgMax,
            LoadImageUrl,
            LoadImageUrlBytes,
            GrayscaleImage,
            PresignedAwsUrl,
            AppendConstString,
            AppendStrings,
            PrependConstString,
            ConcatStringAggregation,
            SumAggregation,
            MeanAggregation,
            MinAggregation,
            MaxAggregation,
            MedianAggregation,
            CountAggregation,
            CountDistinctAggregation,
            StdAggregation,
            VarianceAggregation,
            PercentileAggregation,
            JsonPath,
            Clip,
            ArrayContains,
            ArrayContainsAny,
            ArrayAtIndex,
            OllamaEmbedding,
            PolarsMapRowTransformation,
            LoadFeature,
            FormatStringTransformation,
            ListDotProduct,
        ]:
            self.add(tran_type)

    def add(self, transformation: type[Transformation]) -> None:
        self.types[transformation.name] = transformation

    @classmethod
    def shared(cls) -> SupportedTransformations:
        if cls._shared:
            return cls._shared
        cls._shared = SupportedTransformations()
        return cls._shared


@dataclass
class Expression(Codable):
    """
    A structure that makes it easy to encode column transformations.

    This can also contain references to columns or literal values.

    Therefore, making it possible to nest multiple levels of transformations in one expression.

    E.g.: ((a + b) ** 2) > 20
    """

    column: str | None = field(default=None)
    transformation: Transformation | None = field(default=None)
    literal: LiteralValue | None = field(default=None)

    def needed_columns(self) -> list[str]:
        if self.column is not None:
            return [self.column]
        elif self.transformation:
            return self.transformation.needed_columns()
        return []

    def __and__(self, other: Expression) -> Expression:
        return Expression(
            transformation=BinaryTransformation(left=self, right=other, operator="and")
        )

    def to_spark(self) -> Column | None:
        from pyspark.sql.functions import col, lit

        if self.column:
            return col(self.column)
        if self.literal:
            return lit(self.literal.python_value)
        if self.transformation and isinstance(self.transformation, SparkExpression):
            return self.transformation.spark_col()
        return None

    def to_polars(self) -> pl.Expr | None:
        if self.column:
            return pl.col(self.column)
        if self.literal:
            return pl.lit(self.literal.python_value)
        if self.transformation and isinstance(
            self.transformation, PolarsExprTransformation
        ):
            return self.transformation.polars_expr()
        return None

    def to_glot(self) -> exp.Expression | None:
        if self.column:
            return exp.column(self.column)
        if self.literal:
            val = self.literal.python_value
            if isinstance(val, str):
                return exp.Literal.string(val)
            else:
                return exp.Literal.number(val)
        if self.transformation and isinstance(
            self.transformation, GlotExprTransformation
        ):
            return self.transformation.to_glot()
        return None

    @staticmethod
    def from_value(value: Any) -> Expression:
        from aligned.compiler.feature_factory import FeatureFactory
        from aligned.schemas.derivied_feature import DerivedFeature, Feature

        if isinstance(value, Expression):
            return value

        if isinstance(value, FeatureFactory):
            if value.transformation:
                return Expression(transformation=value.transformation.compile())

            return Expression(column=value.name)

        if isinstance(value, pl.Expr):
            from aligned.polars_to_spark import polars_to_expression

            exp = polars_to_expression(value)
            assert (
                exp is not None
            ), f"Unable to transform polars expression '{value}' to aligned expression."
            return exp

        if isinstance(value, (Feature, DerivedFeature)):
            return value.to_expression()

        return Expression(literal=LiteralValue.from_value(value))


BinaryOperators = Literal[
    "add",
    "sub",
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "mul",
    "div",
    "or",
    "and",
    "pow",
    "mod",
    "xor",
    "list_contains",
    "isin",
    "floor_div",
    "min",
    "max",
    "concat",
    "str_contains",
    "str_starts_with",
    "str_ends_with",
    "str_split",
    "str_find",
]


@dataclass
class BinaryTransformation(
    Transformation, PolarsExprTransformation, SparkExpression, GlotExprTransformation
):
    left: Expression
    right: Expression

    operator: BinaryOperators
    dtype: FeatureType = FeatureType.string()
    name: str = "binary"

    def needed_columns(self) -> list[str]:
        all = self.left.needed_columns()
        all.extend(self.right.needed_columns())
        return all

    def to_glot(self) -> exp.Expression:
        left = self.left.to_glot()
        right = self.right.to_glot()

        assert left is not None
        assert right is not None

        match self.operator:
            case "neq":
                return exp.NEQ(this=left, expression=right)
            case "eq":
                return exp.EQ(this=left, expression=right)
            case "gte":
                return exp.GTE(this=left, expression=right)
            case "gt":
                return exp.GT(this=left, expression=right)
            case "lte":
                return exp.LTE(this=left, expression=right)
            case "lt":
                return exp.LT(this=left, expression=right)
            case "add":
                return exp.Add(this=left, expression=right)
            case "sub":
                return exp.Sub(this=left, expression=right)
            case "mul":
                return exp.Mul(this=left, expression=right)
            case "div":
                return exp.Div(this=left, expression=right)
            case "mod":
                return exp.Mod(this=left, expression=right)
            case "xor":
                return exp.BitwiseXor(this=left, expression=right)
            case "and":
                return exp.And(this=left, expression=right)
            case "or":
                return exp.Or(this=left, expression=right)
            case "pow":
                return exp.Pow(this=left, expression=right)
            case "str_contains":
                return exp.Anonymous(this="CONTAINS", expressions=[left, right])
            case "list_contains":
                return exp.Anonymous(this="ARRAY_CONTAINS", expressions=[left, right])
            case "isin":
                return exp.In(this=left, expressions=[right])
            case "floor_div":
                # SQLglot doesn't have floor division, simulate with FLOOR(a / b)
                return exp.Anonymous(
                    this="FLOOR", expressions=[exp.Div(this=left, expression=right)]
                )
            case "min":
                return exp.Anonymous(this="LEAST", expressions=[left, right])
            case "max":
                return exp.Anonymous(this="GREATEST", expressions=[left, right])
            case "concat":
                return exp.Concat(expressions=[left, right])
            case "str_starts_with":
                return exp.Anonymous(this="STARTS_WITH", expressions=[left, right])
            case "str_ends_with":
                return exp.Anonymous(this="ENDS_WITH", expressions=[left, right])
            case "str_split":
                return exp.Anonymous(this="SPLIT", expressions=[left, right])
            case "str_find":
                return exp.Anonymous(this="POSITION", expressions=[right, left])
            case _:
                raise ValueError(f"Unable to format '{self.operator}' for {self}")

    def polars_expr(self) -> pl.Expr | None:
        left_exp = self.left.to_polars()
        right_exp = self.right.to_polars()

        if left_exp is not None and right_exp is not None:
            return self._polars_expr(left_exp, right_exp)

        return None

    def spark_col(self) -> Column | None:
        left = self.left.to_spark()
        right = self.right.to_spark()

        if left is None or right is None:
            return None

        match self.operator:
            case "neq":
                return left != right
            case "eq":
                return left == right
            case "gte":
                return left >= right
            case "gt":
                return left > right
            case "lte":
                return left <= right
            case "lt":
                return left < right
            case "add":
                return left + right
            case "sub":
                return left - right
            case "mul":
                return left * right
            case "div":
                return left / right
            case "mod":
                return left % right
            case "xor":
                return left.bitwiseXOR(right)
            case "and":
                return left & right
            case "or":
                return left | right
            case "pow":
                return left**right
            case "str_contains":
                return left.contains(right)
            case "list_contains":
                import pyspark.sql.functions as F

                return F.array_contains(left, right)
            case "isin":
                return left.isin(right)
            case "floor_div":
                # Spark doesn't have floor division operator, but we can simulate it
                return (left / right).cast("integer")
            case "min":
                import pyspark.sql.functions as F

                return F.least(left, right)
            case "max":
                import pyspark.sql.functions as F

                return F.greatest(left, right)
            case "concat":
                import pyspark.sql.functions as F

                return F.concat(left, right)
            case "str_starts_with":
                return left.startswith(right)
            case "str_ends_with":
                return left.endswith(right)
            case "str_split":
                import pyspark.sql.functions as F

                return F.split(left, right)
            case "str_find":
                import pyspark.sql.functions as F

                assert (
                    self.right.literal is not None
                ), "Needed a python literal got None"
                return F.locate(self.right.literal.python_value, left) - 1
            case _:
                raise ValueError(f"Unable to format '{self.operator}' for {self}")

    def _polars_expr(self, left: pl.Expr, right: pl.Expr) -> pl.Expr:
        match self.operator:
            case "add":
                return left + right
            case "sub":
                return left - right
            case "eq":
                return left == right
            case "neq":
                return left != right
            case "gt":
                return left > right
            case "gte":
                return left >= right
            case "lt":
                return left < right
            case "lte":
                return left <= right
            case "mul":
                return left * right
            case "div":
                return left / right
            case "or":
                return left | right
            case "and":
                return left & right
            case "pow":
                return left.pow(right)
            case "mod":
                return left.mod(right)
            case "xor":
                return left.xor(right)
            case "str_contains":
                return left.str.contains(right)
            case "list_contains":
                return left.list.contains(right)
            case "isin":
                return left.is_in(right)
            case "floor_div":
                return left.floordiv(right)
            case "min":
                return pl.min_horizontal([left, right])
            case "max":
                return pl.max_horizontal([left, right])
            case "concat":
                return pl.concat_str([left, right])
            case "str_starts_with":
                return left.str.starts_with(right)
            case "str_ends_with":
                return left.str.ends_with(right)
            case "str_split":
                return left.str.split(right)
            case "str_find":
                return left.str.find(right)
            case _:
                raise ValueError(f"Unable to compute {self.operator}")

    def pandas_op(self, left: pd.Series, right: pd.Series) -> pd.Series:
        match self.operator:
            case "add":
                return left + right
            case "sub":
                return left - right
            case "eq":
                return left == right
            case "neq":
                return left != right
            case "gt":
                return left > right
            case "gte":
                return left >= right
            case "lt":
                return left < right
            case "lte":
                return left <= right
            case "mul":
                return left * right
            case "div":
                return left / right
            case "or":
                return left | right
            case "and":
                return left & right
            case "pow":
                return left**right
            case "mod":
                return left.mod(right)
            case "xor":
                return left ^ right
            case "str_contains":
                return left.str.contains(right)
            case "list_contains":
                pl_left = pl.Series(left)
                pl_right = pl.Series(right)
                return pl_left.list.contains(pl_right).to_pandas()
            case "isin":
                return left.isin(right)
            case "floor_div":
                return left // right
            case "min":
                return np.minimum(left, right)  # type: ignore
            case "max":
                return np.maximum(left, right)  # type: ignore
            case "concat":
                return left.astype(str) + right.astype(str)
            case "str_starts_with":
                return left.str.startswith(right)
            case "str_ends_with":
                return left.str.endswith(right)
            case "str_split":
                return left.str.split(right)
            case "str_find":
                return left.str.find(right)
            case _:
                raise ValueError(f"Unable to compute {self.operator}")

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        exp = self.polars_expr()
        if exp is not None:
            return exp

        left_exp = self.left.to_polars()
        right_exp = self.right.to_polars()

        left_col = "_aligned_left"
        right_col = "_aligned_right"

        if left_exp is None and self.left.transformation:
            out = await self.left.transformation.transform_polars(df, left_col, store)
            if isinstance(out, pl.Expr):
                left_exp = out
            else:
                df = out
                left_exp = pl.col(left_col)

        if right_exp is None and self.right.transformation:
            out = await self.right.transformation.transform_polars(df, right_col, store)
            if isinstance(out, pl.Expr):
                right_exp = out
            else:
                df = out
                right_exp = pl.col(right_col)

        assert left_exp is not None
        assert right_exp is not None

        new_exp = self._polars_expr(left_exp, right_exp)
        return df.with_columns(new_exp).select(pl.exclude([left_col, right_col]))

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        left_series = None
        right_series = None

        if self.left.column:
            left_series = df[self.left.column]
        elif self.left.literal:
            left_series = self.left.literal.python_value
        else:
            assert self.left.transformation
            left_series = await self.left.transformation.transform_pandas(df, store)

        if self.right.column:
            right_series = df[self.right.column]
        elif self.right.literal:
            right_series = self.right.literal.python_value
        else:
            assert self.right.transformation
            right_series = await self.right.transformation.transform_pandas(df, store)

        assert left_series is not None
        assert right_series is not None

        return self.pandas_op(left_series, right_series)  # type: ignore


UnaryFunction = Literal[
    "is_null",
    "is_not_null",
    "is_nan",
    "is_not_nan",
    "is_finite",
    "is_infinite",
    "not",
    "floor",
    "ceil",
    "round",
    "abs",
    "sqrt",
    "log",
    "log10",
    "exp",
    "sign",
    "sin",
    "cos",
    "tan",
    "cot",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "degrees",
    "radians",
    "log1p",
    "str_len_char",
    "str_to_upper",
    "str_to_lower",
]


@dataclass
class UnaryTransformation(Transformation, InnerTransformation, GlotExprTransformation):
    inner: Expression
    func: UnaryFunction
    name: str = "unary"
    dtype: FeatureType = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return self.inner.needed_columns()

    def to_glot(self) -> exp.Expression:
        inner_exp = self.inner.to_glot()
        assert inner_exp is not None

        match self.func:
            case "is_null":
                return exp.Anonymous(this="IS NULL", expressions=[inner_exp])
            case "is_not_null":
                return exp.Anonymous(this="IS NOT NULL", expressions=[inner_exp])
            case "is_nan":
                return exp.Anonymous(this="IS_NAN", expressions=[inner_exp])
            case "is_not_nan":
                return exp.Anonymous(this="NOT IS_NAN", expressions=[inner_exp])
            case "is_finite":
                return exp.Anonymous(this="IS_FINITE", expressions=[inner_exp])
            case "is_infinite":
                return exp.Anonymous(this="IS_INFINITE", expressions=[inner_exp])
            case "not":
                return exp.Not(this=inner_exp)
            case "floor":
                return exp.Anonymous(this="FLOOR", expressions=[inner_exp])
            case "ceil":
                return exp.Anonymous(this="CEIL", expressions=[inner_exp])
            case "round":
                return exp.Anonymous(this="ROUND", expressions=[inner_exp])
            case "abs":
                return exp.Anonymous(this="ABS", expressions=[inner_exp])
            case "sqrt":
                return exp.Anonymous(this="SQRT", expressions=[inner_exp])
            case "log":
                return exp.Log(this="LOG", expressions=[inner_exp])
            case "log10":
                return exp.Anonymous(this="LOG10", expressions=[inner_exp])
            case "exp":
                return exp.Anonymous(this="EXP", expressions=[inner_exp])
            case "sign":
                return exp.Anonymous(this="SIGN", expressions=[inner_exp])
            case "sin":
                return exp.Anonymous(this="SIN", expressions=[inner_exp])
            case "cos":
                return exp.Anonymous(this="COS", expressions=[inner_exp])
            case "tan":
                return exp.Anonymous(this="TAN", expressions=[inner_exp])
            case "cot":
                return exp.Anonymous(this="COT", expressions=[inner_exp])
            case "arcsin":
                return exp.Anonymous(this="ASIN", expressions=[inner_exp])
            case "arccos":
                return exp.Anonymous(this="ACOS", expressions=[inner_exp])
            case "arctan":
                return exp.Anonymous(this="ATAN", expressions=[inner_exp])
            case "sinh":
                return exp.Anonymous(this="SINH", expressions=[inner_exp])
            case "cosh":
                return exp.Anonymous(this="COSH", expressions=[inner_exp])
            case "tanh":
                return exp.Anonymous(this="TANH", expressions=[inner_exp])
            case "arcsinh":
                return exp.Anonymous(this="ASINH", expressions=[inner_exp])
            case "arccosh":
                return exp.Anonymous(this="ACOSH", expressions=[inner_exp])
            case "arctanh":
                return exp.Anonymous(this="ATANH", expressions=[inner_exp])
            case "degrees":
                return exp.Anonymous(this="DEGREES", expressions=[inner_exp])
            case "radians":
                return exp.Anonymous(this="RADIANS", expressions=[inner_exp])
            case "log1p":
                return exp.Anonymous(this="LOG1P", expressions=[inner_exp])
            case "str_len_char":
                return exp.Anonymous(this="LENGTH", expressions=[inner_exp])
            case "str_to_lower":
                return exp.Anonymous(this="LOWER", expressions=[inner_exp])
            case "str_to_upper":
                return exp.Anonymous(this="UPPER", expressions=[inner_exp])
            case _:
                raise ValueError(
                    f"Unary function '{self.func}' not supported for SQLglot"
                )

    def polars_expr_from(self, inner: pl.Expr) -> pl.Expr:
        match self.func:
            case "is_null":
                return inner.is_null()
            case "is_not_null":
                return inner.is_not_null()
            case "is_nan":
                return inner.is_nan()
            case "is_not_nan":
                return inner.is_not_nan()
            case "is_finite":
                return inner.is_finite()
            case "is_infinite":
                return inner.is_infinite()
            case "not":
                return inner.not_()
            case "floor":
                return inner.floor()
            case "ceil":
                return inner.ceil()
            case "round":
                return inner.round()
            case "abs":
                return inner.abs()
            case "sqrt":
                return inner.sqrt()
            case "log":
                return inner.log()
            case "log10":
                return inner.log10()
            case "exp":
                return inner.exp()
            case "sign":
                return inner.sign()
            case "sin":
                return inner.sin()
            case "cos":
                return inner.cos()
            case "tan":
                return inner.tan()
            case "cot":
                return inner.cot()
            case "arcsin":
                return inner.arcsin()
            case "arccos":
                return inner.arccos()
            case "arctan":
                return inner.arctan()
            case "sinh":
                return inner.sinh()
            case "cosh":
                return inner.cosh()
            case "tanh":
                return inner.tanh()
            case "arcsinh":
                return inner.arcsinh()
            case "arccosh":
                return inner.arccosh()
            case "arctanh":
                return inner.arctanh()
            case "degrees":
                return inner.degrees()
            case "radians":
                return inner.radians()
            case "log1p":
                return inner.log1p()
            case "str_len_char":
                return inner.str.len_chars()
            case "str_to_lower":
                return inner.str.to_lowercase()
            case "str_to_upper":
                return inner.str.to_uppercase()
            case _:
                raise ValueError(f"Unary function '{self.func}' not supported")

    def pandas_tran(self, column: pd.Series) -> pd.Series:
        match self.func:
            case "is_null":
                return column.isnull()
            case "is_not_null":
                return column.notnull()
            case "is_nan":
                return column.isna()
            case "is_not_nan":
                return ~column.isna()
            case "is_finite":
                return np.isfinite(column)  # type: ignore
            case "is_infinite":
                return np.isinf(column)  # type: ignore
            case "not":
                return ~column
            case "floor":
                return np.floor(column)  # type: ignore
            case "ceil":
                return np.ceil(column)  # type: ignore
            case "round":
                return np.round(column)  # type: ignore
            case "abs":
                return np.abs(column)  # type: ignore
            case "sqrt":
                return np.sqrt(column)  # type: ignore
            case "log":
                return np.log(column)  # type: ignore
            case "log10":
                return np.log10(column)  # type: ignore
            case "exp":
                return np.exp(column)  # type: ignore
            case "sign":
                return np.sign(column)  # type: ignore
            case "sin":
                return np.sin(column)  # type: ignore
            case "cos":
                return np.cos(column)  # type: ignore
            case "tan":
                return np.tan(column)  # type: ignore
            case "cot":
                return 1 / np.tan(column)  # type: ignore
            case "arcsin":
                return np.arcsin(column)  # type: ignore
            case "arccos":
                return np.arccos(column)  # type: ignore
            case "arctan":
                return np.arctan(column)  # type: ignore
            case "sinh":
                return np.sinh(column)  # type: ignore
            case "cosh":
                return np.cosh(column)  # type: ignore
            case "tanh":
                return np.tanh(column)  # type: ignore
            case "arcsinh":
                return np.arcsinh(column)  # type: ignore
            case "arccosh":
                return np.arccosh(column)  # type: ignore
            case "arctanh":
                return np.arctanh(column)  # type: ignore
            case "degrees":
                return np.degrees(column)  # type: ignore
            case "radians":
                return np.radians(column)  # type: ignore
            case "log1p":
                return np.log1p(column)  # type: ignore
            case _:
                raise ValueError(
                    f"Unary function '{self.func}' not supported in pandas"
                )

    def spark_col_from(self, inner: Column) -> Column | None:
        import pyspark.sql.functions as F

        match self.func:
            case "is_null":
                return inner.isNull()
            case "is_not_null":
                return inner.isNotNull()
            case "is_nan":
                return F.isnan(inner)
            case "is_not_nan":
                return ~F.isnan(inner)
            case "is_finite":
                return ~(F.isnan(inner) | F.isinf(inner))  # type: ignore
            case "is_infinite":
                return F.isinf(inner)  # type: ignore
            case "not":
                return ~inner
            case "floor":
                return F.floor(inner)
            case "ceil":
                return F.ceil(inner)
            case "round":
                return F.round(inner)
            case "abs":
                return F.abs(inner)
            case "sqrt":
                return F.sqrt(inner)
            case "log":
                return F.log(inner)
            case "log10":
                return F.log10(inner)
            case "exp":
                return F.exp(inner)
            case "sign":
                return F.signum(inner)
            case "sin":
                return F.sin(inner)
            case "cos":
                return F.cos(inner)
            case "tan":
                return F.tan(inner)
            case "cot":
                return F.cot(inner)
            case "arcsin":
                return F.asin(inner)
            case "arccos":
                return F.acos(inner)
            case "arctan":
                return F.atan(inner)
            case "sinh":
                return F.sinh(inner)
            case "cosh":
                return F.cosh(inner)
            case "tanh":
                return F.tanh(inner)
            case "arcsinh":
                # Spark doesn't have built-in arcsinh, using formula: log(x + sqrt(x^2 + 1))
                return F.log(inner + F.sqrt(inner * inner + F.lit(1)))
            case "arccosh":
                # Spark doesn't have built-in arccosh, using formula: log(x + sqrt(x^2 - 1))
                return F.log(inner + F.sqrt(inner * inner - F.lit(1)))
            case "arctanh":
                # Spark doesn't have built-in arctanh, using formula: 0.5 * log((1 + x) / (1 - x))
                return F.lit(0.5) * F.log((F.lit(1) + inner) / (F.lit(1) - inner))
            case "degrees":
                return F.degrees(inner)
            case "radians":
                return F.radians(inner)
            case "log1p":
                return F.log1p(inner)
            case _:
                raise ValueError(f"Unary function '{self.func}' not supported in Spark")

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        exp = self.polars_expr()
        if exp is not None:
            return exp

        inner_col = "_aligned_inner"
        assert self.inner.transformation

        inner = await self.inner.transformation.transform_polars(df, inner_col, store)

        if isinstance(inner, pl.Expr):
            return self.polars_expr_from(inner)

        return inner.with_columns(
            self.polars_expr_from(pl.col(inner_col)).alias(alias)
        ).select(pl.exclude(inner_col))

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        if self.inner.column:
            series = df[self.inner.column]
        elif self.inner.literal:
            series = self.inner.literal.python_value
        else:
            assert self.inner.transformation
            series = await self.inner.transformation.transform_pandas(df, store)

        return self.pandas_tran(series)  # type: ignore

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            transformation=UnaryTransformation(
                inner=Expression(column="x"), func="abs"
            ),
            input={"x": [-1.5, 2.3, -3.7, None]},
            output=[1.5, 2.3, 3.7, None],
        )


@dataclass
class PolarsMapRowTransformation(Transformation):
    """
    This will encode a custom method, that is not a lambda function
    Threfore, we will stort the actual code, and dynamically load it on runtime.

    This is unsafe, but will remove the ModuleImportError for custom methods
    """

    code: str
    function_name: str
    dtype: FeatureType
    name: str = "pol_map_row"

    def needed_columns(self) -> list[str]:
        return []

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return (
            await self.transform_polars(pl.from_pandas(df).lazy(), "value", store)
        ).collect()["value"]  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        if self.function_name not in locals():
            exec(self.code)

        loaded = locals()[self.function_name]

        polars_df = df.collect()
        new_rows = []

        for row in polars_df.to_dicts():
            if asyncio.iscoroutinefunction(loaded):
                row[alias] = await loaded(row, store)
            else:
                row[alias] = loaded(row, store)
            new_rows.append(row)

        return pl.DataFrame(new_rows).lazy()


@dataclass
class PandasFunctionTransformation(Transformation):
    """
    This will encode a custom method, that is not a lambda function
    Threfore, we will stort the actual code, and dynamically load it on runtime.

    This is unsafe, but will remove the ModuleImportError for custom methods
    """

    code: str
    function_name: str
    dtype: FeatureType
    name: str = "pandas_code_tran"

    def needed_columns(self) -> list[str]:
        return []

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        if self.function_name not in locals():
            exec(self.code)

        loaded = locals()[self.function_name]
        if asyncio.iscoroutinefunction(loaded):
            return await loaded(df, store)
        else:
            return loaded(df, store)

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        pandas_df = df.collect().to_pandas()
        if self.function_name not in locals():
            exec(self.code)

        loaded = locals()[self.function_name]
        if asyncio.iscoroutinefunction(loaded):
            pandas_df[alias] = await loaded(pandas_df, store)
        else:
            pandas_df[alias] = loaded(pandas_df, store)

        return pl.from_pandas(pandas_df).lazy()

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            transformation=PandasFunctionTransformation(
                code='async def test(df, store):\n    return df["a"] + df["b"]',
                function_name="test",
                dtype=FeatureType.int32(),
            ),
            input={
                "a": [1, 2, 3, 4, 5],
                "b": [1, 2, 3, 4, 5],
            },
            output=[2, 4, 6, 8, 10],
        )


@dataclass
class PandasLambdaTransformation(Transformation):
    method: bytes
    code: str
    dtype: FeatureType
    name: str = "pandas_lambda_tran"

    def needed_columns(self) -> list[str]:
        return []

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        import asyncio

        import dill

        loaded = dill.loads(self.method)
        if asyncio.iscoroutinefunction(loaded):
            return await loaded(df, store)
        else:
            return loaded(df, store)

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        import dill

        pandas_df = df.collect().to_pandas()
        loaded = dill.loads(self.method)
        if asyncio.iscoroutinefunction(loaded):
            pandas_df[alias] = await loaded(pandas_df, store)
        else:
            pandas_df[alias] = loaded(pandas_df, store)

        return pl.from_pandas(pandas_df).lazy()


@dataclass
class PolarsFunctionTransformation(Transformation):
    """
    This will encode a custom method, that is not a lambda function
    Threfore, we will stort the actual code, and dynamically load it on runtime.

    This is unsafe, but will remove the ModuleImportError for custom methods
    """

    code: str
    function_name: str
    dtype: FeatureType
    name: str = "pandas_code_tran"

    def needed_columns(self) -> list[str]:
        return []

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        polars_df = await self.transform_polars(
            pl.from_pandas(df).lazy(), self.function_name, store
        )
        assert isinstance(polars_df, pl.LazyFrame)
        return polars_df.collect().to_pandas()[self.function_name]  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        if self.function_name not in locals():
            exec(self.code)

        loaded = locals()[self.function_name]
        if asyncio.iscoroutinefunction(loaded):
            return await loaded(df, alias, store)
        else:
            return loaded(df, alias, store)


@dataclass
class PolarsExpression(Transformation, PolarsExprTransformation):
    polars_expression: str
    dtype: FeatureType
    name: str = "polars_expression"

    def polars_expr(self) -> pl.Expr:
        return pl.Expr.deserialize(self.polars_expression.encode(), format="json")

    def needed_columns(self) -> list[str]:
        return self.polars_expr().meta.root_names()

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        pl_df = pl.from_pandas(df)
        pl_df = pl_df.with_columns(
            pl.Expr.deserialize(self.polars_expression.encode(), format="json").alias(
                "polars_tran_column"
            )
        )
        return pl_df["polars_tran_column"].to_pandas()


@dataclass
class PolarsLambdaTransformation(Transformation):
    method: bytes
    code: str
    dtype: FeatureType
    name: str = "polars_lambda_tran"

    def needed_columns(self) -> list[str]:
        return []

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        import dill

        loaded: pl.Expr = dill.loads(self.method)
        pl_df = pl.from_pandas(df)
        pl_df = pl_df.with_columns((loaded).alias("polars_tran_column"))
        return pl_df["polars_tran_column"].to_pandas()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        import dill

        tran = dill.loads(self.method)
        if isinstance(tran, pl.Expr):
            return tran
        else:
            return tran(df, alias, store)


@dataclass
class TimeDifference(Transformation, PsqlTransformation, RedshiftTransformation):
    front: str
    behind: str
    unit: str

    name: str = "time-diff"
    dtype: FeatureType = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.front, self.behind]

    def __init__(self, front: str, behind: str, unit: str = "s") -> None:
        self.front = front
        self.behind = behind
        self.unit = unit

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.front].isna() | df[self.behind].isna()),
            transformation=lambda dfv: (dfv[self.front] - dfv[self.behind])
            / np.timedelta64(1, self.unit),  # type: ignore
        )

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return df.with_columns(
            (pl.col(self.front) - pl.col(self.behind)).dt.total_seconds().alias(alias)
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            TimeDifference(front="x", behind="y"),
            input={
                "x": [
                    datetime.fromtimestamp(1),
                    datetime.fromtimestamp(2),
                    datetime.fromtimestamp(0),
                    None,
                    datetime.fromtimestamp(1),
                ],
                "y": [
                    datetime.fromtimestamp(1),
                    datetime.fromtimestamp(0),
                    datetime.fromtimestamp(2),
                    datetime.fromtimestamp(1),
                    None,
                ],
            },
            output=[0, 2, -2, nan, nan],
        )

    def as_psql(self) -> str:
        return f"DATEDIFF('sec', {self.behind}, {self.front})"


@dataclass
class ToNumerical(Transformation, PolarsExprTransformation):
    key: str

    name: str = "to-num"
    dtype: FeatureType = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    def __init__(self, key: str) -> None:
        self.key = key

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        from pandas import to_numeric

        return to_numeric(df[self.key], errors="coerce")  # type: ignore

    def polars_expr(self) -> pl.Expr:
        return pl.col(self.key).cast(pl.Float64)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            ToNumerical("x"),
            input={"x": ["1", "0", "10.5", None, "-20"]},
            output=[1, 0, 10.5, None, -20],
        )


@dataclass
class DateComponent(Transformation, PolarsExprTransformation, GlotExprTransformation):
    key: str
    component: str

    name: str = "date-component"
    dtype: FeatureType = FeatureType.int32()

    def needed_columns(self) -> list[str]:
        return [self.key]

    def __init__(self, key: str, component: str) -> None:
        self.key = key
        self.component = component

    def to_glot(self) -> exp.Expression:
        col_exp = exp.column(self.key)

        # Map component to SQL date/time extraction functions
        match self.component:
            case "day":
                return exp.Anonymous(this="DAY", expressions=[col_exp])
            case "hour":
                return exp.Anonymous(this="HOUR", expressions=[col_exp])
            case "minute":
                return exp.Anonymous(this="MINUTE", expressions=[col_exp])
            case "second":
                return exp.Anonymous(this="SECOND", expressions=[col_exp])
            case "month":
                return exp.Anonymous(this="MONTH", expressions=[col_exp])
            case "year":
                return exp.Anonymous(this="YEAR", expressions=[col_exp])
            case "quarter":
                return exp.Anonymous(this="QUARTER", expressions=[col_exp])
            case "week":
                return exp.Anonymous(this="WEEK", expressions=[col_exp])
            case "weekday":
                return exp.Anonymous(this="DAYOFWEEK", expressions=[col_exp])
            case "dayofweek":
                return exp.Anonymous(this="DAYOFWEEK", expressions=[col_exp])
            case "epoch":
                return exp.Anonymous(this="UNIX_TIMESTAMP", expressions=[col_exp])
            case _:
                # For unsupported components, fall back to generic EXTRACT function
                return exp.Anonymous(
                    this="EXTRACT",
                    expressions=[exp.Literal.string(self.component.upper()), col_exp],
                )

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna()),  # type: ignore
            transformation=lambda dfv: getattr(dfv[self.key].dt, self.component),  # type: ignore
        )

    def polars_expr(self) -> pl.Expr:
        col = pl.col(self.key).cast(pl.Datetime).dt
        match self.component:
            case "day":
                expr = col.day()
            case "days":
                expr = col.ordinal_day()
            case "epoch":
                expr = col.epoch()
            case "hour":
                expr = col.hour()
            case "hours":
                expr = col.total_hours()
            case "iso_year":
                expr = col.iso_year()
            case "microsecond":
                expr = col.microsecond()
            case "microseconds":
                expr = col.total_microseconds()
            case "millisecond":
                expr = col.millisecond()
            case "milliseconds":
                expr = col.total_milliseconds()
            case "minute":
                expr = col.minute()
            case "minutes":
                expr = col.total_minutes()
            case "month":
                expr = col.month()
            case "nanosecond":
                expr = col.nanosecond()
            case "nanoseconds":
                expr = col.total_nanoseconds()
            case "ordinal_day":
                expr = col.ordinal_day()
            case "quarter":
                expr = col.quarter()
            case "second":
                expr = col.second()
            case "seconds":
                expr = col.total_seconds()
            case "week":
                expr = col.week()
            case "weekday":
                expr = col.weekday()
            case "year":
                expr = col.year()
            case "dayofweek":
                expr = col.weekday()
            case _:
                raise NotImplementedError(
                    f"Date component {self.component} is not implemented. Maybe setup a PR and contribute?"
                )
        return expr

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            DateComponent(key="x", component="hour"),
            input={
                "x": [
                    datetime.fromisoformat(value) if value else None
                    for value in [
                        "2022-04-02T20:20:50",
                        None,
                        "2022-02-20T23:20:50",
                        "1993-04-02T01:20:50",
                    ]
                ]
            },
            output=[20, None, 23, 1],
        )


@dataclass
class ArrayAtIndex(Transformation, PolarsExprTransformation):
    """Checks if an array contains a value

    some_array = List(String())
    contains_a_char = some_array.contains("a")
    """

    key: str
    index: int

    name: str = "array_at_index"
    dtype: FeatureType = FeatureType.boolean()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return pl.Series(df[self.key]).list.get(self.index).to_pandas()

    def polars_expr(self) -> pl.Expr:
        return pl.col(self.key).list.get(self.index)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            ArrayContains("x", LiteralValue.from_value("test")),
            input={"x": [["Hello", "test"], ["nah"], ["test", "espania", None]]},
            output=[True, False, True],
        )


@dataclass
class ArrayContainsAny(Transformation, PolarsExprTransformation):
    """Checks if an array contains a value

    some_array = List(String())
    contains_char = some_array.contains_any(["a", "b"])
    """

    key: str
    values: LiteralValue

    name: str = "array_contains_any"
    dtype: FeatureType = FeatureType.boolean()

    def needed_columns(self) -> list[str]:
        return [self.key]

    def __init__(self, key: str, values: Any | LiteralValue) -> None:
        self.key = key
        if isinstance(values, LiteralValue):
            self.values = values
        else:
            self.values = LiteralValue.from_value(values)

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        vals = self.values.python_value
        return (
            pl.Series(df[self.key])
            .list.eval(pl.element().is_in(vals))
            .list.any()
            .to_pandas()
        )

    def polars_expr(self) -> pl.Expr:
        vals = self.values.python_value
        return pl.col(self.key).list.eval(pl.element().is_in(vals)).list.any()

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            ArrayContainsAny("x", LiteralValue.from_value(["test", "nah"])),
            input={"x": [["Hello", "test"], ["nah"], ["espania", None]]},
            output=[True, True, False],
        )


@dataclass
class ArrayContains(Transformation, PolarsExprTransformation):
    """Checks if an array contains a value

    some_array = List(String())
    contains_a_char = some_array.contains("a")
    """

    key: str
    value: LiteralValue | str

    name: str = "array_contains"
    dtype: FeatureType = FeatureType.boolean()

    def needed_columns(self) -> list[str]:
        return self.polars_expr().meta.root_names()

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        if isinstance(self.value, str):
            pdf = pl.DataFrame({self.key: df[self.key], self.value: df[self.value]})
            return pdf.select(
                output=pl.col(self.key).list.contains(pl.col(self.value))
            )["output"].to_pandas()
        else:
            return (
                pl.Series(df[self.key])
                .list.contains(self.value.python_value)
                .to_pandas()
            )

    def polars_expr(self) -> pl.Expr:
        if isinstance(self.value, str):
            return pl.col(self.key).list.contains(pl.col(self.value))
        else:
            return pl.col(self.key).list.contains(self.value.python_value)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            ArrayContains("x", LiteralValue.from_value("test")),
            input={"x": [["Hello", "test"], ["nah"], ["test", "espania", None]]},
            output=[True, False, True],
        )


@dataclass
class Contains(Transformation, PolarsExprTransformation, GlotExprTransformation):
    """Checks if a string value contains another string

    some_string = String()
    contains_a_char = some_string.contains("a")
    """

    key: str
    value: str

    name: str = "contains"
    dtype: FeatureType = FeatureType.boolean()

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    def needed_columns(self) -> list[str]:
        return self.polars_expr().meta.root_names()

    def to_glot(self) -> exp.Expression:
        col_exp = exp.column(self.key)
        value_literal = exp.Literal.string(self.value)
        # Use CONTAINS or LIKE depending on SQL dialect preference
        return exp.Anonymous(this="CONTAINS", expressions=[col_exp, value_literal])

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna()),  # type: ignore
            transformation=lambda dfv: dfv[self.key]
            .astype("str")
            .str.contains(self.value),  # type: ignore
        )

    def polars_expr(self) -> pl.Expr:
        return pl.col(self.key).str.contains(self.value)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Contains("x", "es"),
            input={"x": ["Hello", "Test", "nah", "test", "espania", None]},
            output=[False, True, False, True, True, None],
        )


@dataclass
class Ordinal(Transformation):
    key: str
    orders: list[str]

    @property
    def orders_dict(self) -> dict[str, int]:
        return {key: index for index, key in enumerate(self.orders)}

    name: str = "ordinal"
    dtype: FeatureType = FeatureType.int32()

    def __init__(self, key: str, orders: list[str]) -> None:
        self.key = key
        self.orders = orders

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return df[self.key].map(self.orders_dict)  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        mapper = pl.DataFrame(
            {self.key: list(self.orders), alias: list(range(0, len(self.orders)))}
        )
        return df.join(mapper.lazy(), on=self.key, how="left")

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Ordinal("x", ["a", "b", "c", "d"]),
            input={"x": ["a", "b", "a", None, "d", "p"]},
            output=[0, 1, 0, None, 3, None],
        )


@dataclass
class ReplaceStrings(Transformation):
    key: str
    values: list[tuple[str, str]]

    name: str = "replace"
    dtype: FeatureType = FeatureType.string()

    def __init__(self, key: str, values: list[tuple[str, str]]) -> None:
        self.key = key
        self.values = values

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        temp_df = df[self.key].copy()
        mask = ~(df[self.key].isna() | df[self.key].isnull())
        temp_df.loc[~mask] = np.nan
        for k, v in self.values:
            temp_df.loc[mask] = temp_df.loc[mask].str.replace(k, v, regex=True)

        return temp_df  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        collected = df.collect()
        pandas_column = collected.select(self.key).to_pandas()
        transformed = await self.transform_pandas(pandas_column, store)
        return collected.with_columns(pl.Series(transformed).alias(alias)).lazy()


@dataclass
class IsIn(Transformation, PolarsExprTransformation, GlotExprTransformation):
    values: list
    key: Expression

    name = "isin"
    dtype = FeatureType.boolean()

    def needed_columns(self) -> list[str]:
        return self.key.needed_columns()

    def to_glot(self) -> exp.Expression:
        col_exp = self.key.to_glot()
        assert col_exp is not None
        # Create literals for each value in the list
        value_literals = []
        for value in self.values:
            if isinstance(value, str):
                value_literals.append(exp.Literal.string(value))
            else:
                value_literals.append(exp.Literal.number(value))
        return exp.In(this=col_exp, expressions=value_literals)

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        assert self.key.column is not None
        return df[self.key.column].isin(self.values)  # type: ignore

    def polars_expr(self) -> pl.Expr:
        key_exp = self.key.to_polars()
        assert key_exp is not None
        return key_exp.is_in(self.values)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            IsIn(values=["hello", "test"], key=Expression(column="x")),
            input={"x": ["No", "Hello", "hello", "test", "nah", "nehtest"]},
            output=[False, False, True, True, False, False],
        )


@dataclass
class FillNaValuesColumns(Transformation):
    key: str
    fill_key: str
    dtype: FeatureType

    name: str = "fill_missing_key"

    def needed_columns(self) -> list[str]:
        return [self.key, self.fill_key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return df[self.key].fillna(df[self.fill_key])  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        if self.dtype == FeatureType.floating_point():
            return (
                pl.col(self.key)
                .fill_nan(pl.col(self.fill_key))
                .fill_null(pl.col(self.fill_key))
            )

        else:
            return pl.col(self.key).fill_null(pl.col(self.fill_key))

    def should_skip(self, output_column: str, columns: list[str]) -> bool:
        return False

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            FillNaValuesColumns("x", "y", dtype=FeatureType.int32()),
            input={
                "x": [1, 1, None, None, 3, 3, None, 4, 5, None],
                "y": [1, 2, 1, 2, 7, 2, 4, 1, 1, 9],
            },
            output=[1, 1, 1, 2, 3, 3, 4, 4, 5, 9],
        )


@dataclass
class FillNaValues(Transformation, PolarsExprTransformation, GlotExprTransformation):
    key: str
    value: LiteralValue
    dtype: FeatureType

    name: str = "fill_missing"

    def needed_columns(self) -> list[str]:
        return [self.key]

    def to_glot(self) -> exp.Expression:
        col_exp = exp.column(self.key)
        if isinstance(self.value.python_value, str):
            value_literal = exp.Literal.string(self.value.python_value)
        else:
            value_literal = exp.Literal.number(self.value.python_value)

        # Use COALESCE to fill null values - standard SQL function
        return exp.Anonymous(this="COALESCE", expressions=[col_exp, value_literal])

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return df[self.key].fillna(self.value.python_value)  # type: ignore

    def polars_expr(self) -> pl.Expr:
        if self.dtype == FeatureType.floating_point():
            return (
                pl.col(self.key)
                .fill_nan(self.value.python_value)
                .fill_null(self.value.python_value)
            )
        else:
            return pl.col(self.key).fill_null(self.value.python_value)

    def should_skip(self, output_column: str, columns: list[str]) -> bool:
        return False

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            FillNaValues("x", LiteralValue.from_value(3), dtype=FeatureType.int32()),
            input={"x": [1, 1, None, None, 3, 3, None, 4, 5, None]},
            output=[1, 1, 3, 3, 3, 3, 3, 4, 5, 3],
        )


@dataclass
class CopyTransformation(Transformation, PolarsExprTransformation):
    key: str
    dtype: FeatureType

    name: str = "nothing"

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return df[self.key]  # type: ignore

    def polars_expr(self) -> pl.Expr:
        return pl.col(self.key)


@dataclass
class MapArgMax(Transformation):
    column_mappings: dict[str, LiteralValue]
    name = "map_arg_max"

    @property
    def dtype(self) -> FeatureType:  # type: ignore
        return list(self.column_mappings.values())[0].dtype

    def needed_columns(self) -> list[str]:
        return list(self.column_mappings.keys())

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        pl_df = await self.transform_polars(pl.from_pandas(df).lazy(), "feature", store)
        return pl_df.collect().to_pandas()["feature"]  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        expr: pl.Expr = pl.lit(None)

        if len(self.column_mappings) == 1:
            key, value = list(self.column_mappings.items())[0]
            if self.dtype == FeatureType.boolean():
                expr = (
                    pl.when(pl.col(key) > 0.5)
                    .then(value.python_value)
                    .otherwise(not value.python_value)
                )
            elif self.dtype == FeatureType.string():
                expr = (
                    pl.when(pl.col(key) > 0.5)
                    .then(value.python_value)
                    .otherwise(f"not {value.python_value}")
                )
            else:
                expr = (
                    pl.when(pl.col(key) > 0.5)
                    .then(value.python_value)
                    .otherwise(pl.lit(None))
                )
            return expr.alias(alias)
        else:
            features = list(self.column_mappings.keys())
            arg_max_alias = f"{alias}_arg_max"
            array_row_alias = f"{alias}_row"
            mapper = pl.DataFrame(
                {
                    alias: [
                        self.column_mappings[feature].python_value
                        for feature in features
                    ],
                    arg_max_alias: list(range(0, len(features))),
                }
            ).with_columns(pl.col(arg_max_alias).cast(pl.UInt32))
            sub = df.with_columns(
                pl.concat_list(pl.col(features)).alias(array_row_alias)
            ).with_columns(pl.col(array_row_alias).list.arg_max().alias(arg_max_alias))
            return sub.join(mapper.lazy(), on=arg_max_alias, how="left").select(
                pl.exclude([arg_max_alias, array_row_alias])
            )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            MapArgMax(
                {
                    "a_prob": LiteralValue.from_value("a"),
                    "b_prob": LiteralValue.from_value("b"),
                    "c_prob": LiteralValue.from_value("c"),
                }
            ),
            input={
                "a_prob": [0.01, 0.9, 0.25],
                "b_prob": [0.9, 0.05, 0.15],
                "c_prob": [0.09, 0.05, 0.6],
            },
            output=["b", "a", "c"],
        )


@dataclass
class WordVectoriser(Transformation):
    key: str
    model: EmbeddingModel

    name = "word_vectoriser"
    dtype = FeatureType.embedding(768)

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return await self.model.vectorise_pandas(df[self.key])  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return await self.model.vectorise_polars(df, self.key, alias)


@dataclass
class LoadImageUrlBytes(Transformation):
    image_url_key: str

    name = "load_image"
    dtype = FeatureType.binary()

    def needed_columns(self) -> list[str]:
        return [self.image_url_key]

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        import asyncio
        from aligned.sources.local import StorageFileSource

        urls = df.select(self.image_url_key).collect()[self.image_url_key]
        logger.info("Fetching image bytes")
        images = await asyncio.gather(
            *[StorageFileSource(url).read() for url in urls.to_list()]
        )
        logger.info("Loaded all images")
        image_dfs = pl.DataFrame({alias: images})

        return df.with_context(image_dfs.lazy()).select(pl.all())


@dataclass
class LoadImageUrl(Transformation):
    image_url_key: str

    name = "load_image"
    dtype = FeatureType.array()

    def needed_columns(self) -> list[str]:
        return [self.image_url_key]

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        import asyncio
        from io import BytesIO

        import numpy as np
        from PIL import Image

        from aligned.sources.local import StorageFileSource

        urls = df.select(self.image_url_key).collect()[self.image_url_key]

        images = await asyncio.gather(
            *[StorageFileSource(url).read() for url in urls.to_list()]
        )
        data = [np.asarray(Image.open(BytesIO(buffer))) for buffer in images]
        image_dfs = pl.DataFrame({alias: data})
        return df.with_context(image_dfs.lazy()).select(pl.all())


@dataclass
class GrayscaleImage(Transformation):
    image_key: str

    name = "grayscale_image"
    dtype = FeatureType.array()

    def needed_columns(self) -> list[str]:
        return [self.image_key]

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        import numpy as np

        def grayscale(images) -> pl.Series:
            return pl.Series(
                [
                    np.mean(image, axis=2) if len(image.shape) == 3 else image
                    for image in images.to_list()
                ]
            )

        return pl.col(self.image_key).map_batches(grayscale).alias(alias)


@dataclass
class AppendConstString(Transformation, PolarsExprTransformation):
    key: str
    string: str

    name = "append_const_string"
    dtype = FeatureType.string()

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return df[self.key] + self.string

    def polars_expr(self) -> pl.Expr:
        return pl.concat_str(
            [pl.col(self.key).fill_null(""), pl.lit(self.string)], separator=""
        )


@dataclass
class AppendStrings(Transformation, PolarsExprTransformation):
    first_key: str
    second_key: str
    sep: str

    name = "append_strings"
    dtype = FeatureType.string()

    def needed_columns(self) -> list[str]:
        return [self.first_key, self.second_key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return df[self.first_key] + self.sep + df[self.second_key]

    def polars_expr(self) -> pl.Expr:
        return pl.concat_str(
            [
                pl.col(self.first_key).fill_null(""),
                pl.col(self.second_key).fill_null(""),
            ],
            separator=self.sep,
        )


@dataclass
class PrependConstString(Transformation, PolarsExprTransformation):
    string: str
    key: str

    name = "prepend_const_string"
    dtype = FeatureType.string()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return self.string + df[self.key]

    def polars_expr(self) -> pl.Expr:
        return pl.concat_str(
            [pl.lit(self.string), pl.col(self.key).fill_null("")], separator=""
        )


@dataclass
class ConcatStringAggregation(
    Transformation, PsqlTransformation, RedshiftTransformation
):
    key: str
    separator: str = field(default=" ")

    name = "concat_string_agg"
    dtype = FeatureType.string()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        pdf = await self.transform_polars(pl.from_pandas(df).lazy(), self.name, store)
        assert isinstance(pdf, pl.LazyFrame)
        return pdf.collect().to_pandas()[self.name]  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return df.with_columns(
            pl.concat_str(pl.col(self.key), separator=self.separator).alias(alias)
        )

    def as_psql(self) -> str:
        return f"array_to_string(array_agg({self.key}), '{self.separator}')"

    def as_redshift(self) -> str:
        return f'listagg("{self.key}", \'{self.separator}\') within group (order by "{self.key}")'


@dataclass
class SumAggregation(Transformation, PsqlTransformation, RedshiftTransformation):
    key: str

    name = "sum_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return pl.sum(self.key)

    def as_psql(self) -> str:
        return f"SUM({self.key})"


@dataclass
class MeanAggregation(Transformation, PsqlTransformation, RedshiftTransformation):
    key: str

    name = "mean_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).mean()

    def as_psql(self) -> str:
        return f"AVG({self.key})"


@dataclass
class MinAggregation(Transformation, PsqlTransformation, RedshiftTransformation):
    key: str

    name = "min_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).min()

    def as_psql(self) -> str:
        return f"MIN({self.key})"


@dataclass
class MaxAggregation(Transformation, PsqlTransformation, RedshiftTransformation):
    key: str

    name = "max_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).max()

    def as_psql(self) -> str:
        return f"MAX({self.key})"


@dataclass
class CountAggregation(Transformation, PsqlTransformation, RedshiftTransformation):
    key: str

    name = "count_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).count()

    def as_psql(self) -> str:
        return f"COUNT({self.key})"


@dataclass
class CountDistinctAggregation(
    Transformation, PsqlTransformation, RedshiftTransformation
):
    key: str

    name = "count_distinct_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).unique_counts()

    def as_psql(self) -> str:
        return f"COUNT(DISTINCT {self.key})"


@dataclass
class StdAggregation(Transformation, PsqlTransformation, RedshiftTransformation):
    key: str

    name = "std_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).std()

    def as_psql(self) -> str:
        return f"STDDEV({self.key})"


@dataclass
class VarianceAggregation(Transformation, PsqlTransformation, RedshiftTransformation):
    key: str

    name = "var_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).var()

    def as_psql(self) -> str:
        return f"variance({self.key})"


@dataclass
class MedianAggregation(Transformation, PsqlTransformation, RedshiftTransformation):
    key: str

    name = "median_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).median()

    def as_psql(self) -> str:
        return f"percentile_cont(0.5) WITHIN GROUP(ORDER BY {self.key})"


@dataclass
class PercentileAggregation(
    Transformation, PsqlTransformation, RedshiftTransformation, PolarsExprTransformation
):
    key: str
    percentile: float

    name = "percentile_agg"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        raise NotImplementedError()

    def polars_expr(self) -> pl.Expr:
        return pl.col(self.key).quantile(self.percentile)

    def as_psql(self) -> str:
        return f"percentile_cont({self.percentile}) WITHIN GROUP(ORDER BY {self.key})"


@dataclass
class Clip(Transformation, InnerTransformation):
    inner: Expression
    lower: LiteralValue
    upper: LiteralValue

    name = "clip"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return self.inner.needed_columns()

    def pandas_tran(self, column: pd.Series) -> pd.Series:
        return column.clip(lower=self.lower.python_value, upper=self.upper.python_value)  # type: ignore

    def polars_expr_from(self, inner: pl.Expr) -> pl.Expr:
        return inner.clip(
            lower_bound=self.lower.python_value, upper_bound=self.upper.python_value
        )

    def spark_col_from(self, inner: Column) -> Column | None:
        import pyspark.sql.functions as F

        return F.greatest(
            F.least(inner, F.lit(self.upper.python_value)),
            F.lit(self.lower.python_value),
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            transformation=Clip(
                inner=Expression(column="a"),
                lower=LiteralValue.from_value(0),
                upper=LiteralValue.from_value(1),
            ),
            input={"a": [-1, 0.1, 0.9, 2]},
            output=[0, 0.1, 0.9, 1],
        )


@dataclass
class PresignedAwsUrl(Transformation):
    config: AwsS3Config
    key: str

    max_age_seconds: int = field(default=30)

    name = "presigned_aws_url"
    dtype = FeatureType.string()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        from aioaws.s3 import S3Client
        from httpx import AsyncClient

        s3 = S3Client(AsyncClient(), config=self.config.s3_config)
        return df[self.key].apply(
            lambda x: s3.signed_download_url(x, max_age=self.max_age_seconds)
        )  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        from aioaws.s3 import S3Client
        from httpx import AsyncClient

        s3 = S3Client(AsyncClient(), config=self.config.s3_config)

        return df.with_columns(
            pl.col(self.key)
            .map_elements(
                lambda x: s3.signed_download_url(x, max_age=self.max_age_seconds)
            )
            .alias(alias)
        )


@dataclass
class StructField(Transformation):
    key: str
    field: str

    name = "struct_field"
    dtype = FeatureType.string()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        data = pl.from_pandas(df).lazy()
        tran = await self.transform_polars(data, "feature", store)

        if isinstance(tran, pl.LazyFrame):
            return tran.collect().to_pandas()["feature"]  # type: ignore

        return data.select(tran).collect().to_pandas()["feature"]  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        if df.schema[self.key].is_(pl.Utf8):
            return await JsonPath(self.key, f"$.{self.field}").transform_polars(
                df, alias, store
            )
        else:
            return pl.col(self.key).struct.field(self.field).alias(alias)


@dataclass
class OllamaGenerate(Transformation):
    key: str
    model: str
    system: str

    host_env: str | None = None
    name = "ollama_embedding"
    dtype = FeatureType.json()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        from ollama import AsyncClient
        import os

        host = None
        if self.host_env:
            host = os.getenv(self.host_env)

        client = AsyncClient(host=host)

        response = pd.Series([[]] * df.shape[0])

        for index, row in df.iterrows():
            response.iloc[index] = await client.generate(
                model=self.model,
                prompt=row[self.key],  # type: ignore
                system=self.system,
            )

        return response

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        def generate_embedding(values: pl.Series) -> pl.Series:
            from ollama import Client
            import os

            host = None
            if self.host_env:
                host = os.getenv(self.host_env)

            client = Client(host=host)

            return pl.Series(
                [
                    str(
                        client.generate(
                            model=self.model,
                            prompt=value,
                            system=self.system,
                        )
                    )
                    for value in values
                ]
            )

        return pl.col(self.key).map_batches(
            generate_embedding, return_dtype=pl.String()
        )


@dataclass
class OllamaEmbedding(Transformation):
    key: str
    model: str

    host_env: str | None = None
    name = "ollama_embedding"
    dtype = FeatureType.embedding(768)

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        from ollama import AsyncClient
        import os

        host = None
        if self.host_env:
            host = os.getenv(self.host_env)

        client = AsyncClient(host=host)

        response = pd.Series([[]] * df.shape[0])

        for index, row in df.iterrows():
            embedded: dict[str, list] = await client.embeddings(
                self.model,
                row[self.key],  # type: ignore
            )
            response.iloc[index] = embedded["embedding"]

        return response

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        def generate_embedding(values: pl.Series) -> pl.Series:
            from ollama import Client
            import os

            host = None
            if self.host_env:
                host = os.getenv(self.host_env)

            client = Client(host=host)

            values = [
                client.embeddings(self.model, value)["embedding"]
                for value in values  # type: ignore
            ]
            return pl.Series(values)

        return pl.col(self.key).map_batches(
            generate_embedding, return_dtype=pl.List(pl.Float64())
        )


@dataclass
class JsonPath(Transformation, PolarsExprTransformation):
    key: str
    path: str

    name = "json_path"
    dtype = FeatureType.string()

    def needed_columns(self) -> list[str]:
        return [self.key]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        return pl.Series(df[self.key]).str.json_path_match(self.path).to_pandas()

    def polars_expr(self) -> pl.Expr:
        return pl.col(self.key).str.json_path_match(self.path)


@dataclass
class IsBetweenTransformation(
    PolarsExprTransformation, SparkExpression, Transformation
):
    value: Expression
    lower_bound: Expression
    upper_bound: Expression

    name = "is_between"
    dtype: FeatureType = FeatureType.boolean()

    def needed_columns(self) -> list[str]:
        return [
            *self.value.needed_columns(),
            *self.lower_bound.needed_columns(),
            *self.upper_bound.needed_columns(),
        ]

    def pandas_tran(self, column: pd.Series) -> pd.Series:
        import numpy as np

        return np.log(self.base, column)  # type: ignore

    def polars_expr(self) -> pl.Expr:
        val = self.value.to_polars()
        lower = self.lower_bound.to_polars()
        upper = self.upper_bound.to_polars()
        assert val is not None
        assert lower is not None
        assert upper is not None
        return val.is_between(lower, upper)

    def spark_col(self) -> Column:
        val = self.value.to_spark()
        lower = self.lower_bound.to_spark()
        upper = self.upper_bound.to_spark()
        assert val is not None
        assert lower is not None
        assert upper is not None
        return (lower <= val) & (val <= upper)


@dataclass
class CastTransform(Transformation, InnerTransformation, GlotExprTransformation):
    inner: Expression
    dtype: FeatureType

    name = "cast"
    dtype: FeatureType = FeatureType.float32()

    def needed_columns(self) -> list[str]:
        return self.inner.needed_columns()

    def to_glot(self) -> exp.Expression:
        inner_exp = self.inner.to_glot()
        assert inner_exp is not None

        # Map FeatureType to SQL types
        sql_type = None
        if self.dtype == FeatureType.int32():
            sql_type = "INT"
        elif self.dtype == FeatureType.int64():
            sql_type = "BIGINT"
        elif self.dtype == FeatureType.float32():
            sql_type = "FLOAT"
        elif (
            self.dtype == FeatureType.float64()
            or self.dtype == FeatureType.floating_point()
        ):
            sql_type = "DOUBLE"
        elif self.dtype == FeatureType.string():
            sql_type = "VARCHAR"
        elif self.dtype == FeatureType.boolean():
            sql_type = "BOOLEAN"
        else:
            # Default fallback for unsupported types
            sql_type = "VARCHAR"

        return exp.Cast(this=inner_exp, to=sql_type)

    def pandas_tran(self, column: pd.Series) -> pd.Series:
        return column.astype(self.dtype.pandas_type)

    def polars_expr_from(self, inner: pl.Expr) -> pl.Expr:
        return inner.cast(self.dtype.polars_type)

    def spark_col_from(self, inner: Column) -> Column | None:
        return inner.cast(self.dtype.spark_type)


@dataclass
class Log(Transformation, InnerTransformation):
    inner: Expression
    base: float

    name = "log"
    dtype: FeatureType = FeatureType.float32()

    def needed_columns(self) -> list[str]:
        return self.inner.needed_columns()

    def pandas_tran(self, column: pd.Series) -> pd.Series:
        import numpy as np

        return np.log(self.base, column)  # type: ignore

    def polars_expr_from(self, inner: pl.Expr) -> pl.Expr:
        return inner.log(self.base)

    def spark_col_from(self, inner: Column) -> Column | None:
        import pyspark.sql.functions as F

        return F.log(self.base, inner)


@dataclass
class Split(Transformation, InnerTransformation):
    inner: Expression
    separator: str
    name = "split"
    dtype: FeatureType = FeatureType.array(FeatureType.string())

    def needed_columns(self) -> list[str]:
        return self.inner.needed_columns()

    def pandas_tran(self, column: pd.Series) -> pd.Series:
        return column.str.split(self.separator)

    def polars_expr_from(self, inner: pl.Expr) -> pl.Expr:
        return inner.str.split(self.separator)

    def spark_col_from(self, inner: Column) -> Column | None:
        import pyspark.sql.functions as F

        return F.split(inner, pattern=self.separator)


@dataclass
class LoadFeature(Transformation):
    entities: dict[str, str]
    feature: FeatureReference
    explode_key: str | None
    dtype: FeatureType
    name = "load_feature"

    def needed_columns(self) -> list[str]:
        return list(self.entities.values())

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        entities = {}
        for key, df_key in self.entities.items():
            entities[key] = df[df_key]

        values = await store.features_for(
            entities, features=[self.feature.identifier]
        ).to_pandas()
        return values[self.feature.name]  # type: ignore

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        group_keys = []

        if self.explode_key:
            group_keys = ["row_nr"]
            entity_df = df.with_row_index("row_nr").explode(self.explode_key)
        else:
            entity_df = df

        entities = entity_df.rename(
            {df_key: key for key, df_key in self.entities.items()}
        )

        values = (
            await store.features_for(
                entities.collect(), features=[self.feature.identifier]
            )
            .with_subfeatures()
            .to_polars()
        )

        if group_keys:
            values = values.group_by(group_keys).agg(
                [pl.col(col) for col in values.columns if col not in group_keys]
            )

        values = values.select(pl.col(self.feature.name).alias(alias))

        return pl.concat([df, values.lazy()], how="horizontal")


@dataclass
class FormatStringTransformation(Transformation):
    format: str
    keys: list[str]
    name = "format_string"

    def needed_columns(self) -> list[str]:
        return self.keys

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        values = []
        for row in df[self.keys].to_dict(orient="records"):  # type: ignore
            values.append(self.format.format(**row))

        return pd.Series(values)

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        polars_df = df.collect()
        new_rows = []

        for row in polars_df.to_dicts():
            row[alias] = self.format.format(**row)
            new_rows.append(row)

        return pl.DataFrame(new_rows).lazy()


@dataclass
class ListDotProduct(Transformation):
    left: str
    right: str

    name = "list_dot_product"
    dtype = FeatureType.floating_point()

    def needed_columns(self) -> list[str]:
        return [self.left, self.right]

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        pl_df = pl.from_pandas(df)
        res = await self.transform_polars(pl_df.lazy(), "output", store)
        if isinstance(res, pl.Expr):
            return pl_df.with_columns(res.alias("output"))["output"].to_pandas()
        else:
            return res.collect()["output"].to_pandas()

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        polars_version = pl.__version__.split(".")
        if len(polars_version) != 3:
            polars_version = [1, 8, 0]
        else:
            polars_version = [int(num) for num in polars_version]

        if polars_version[0] >= 1 and polars_version[1] >= 8:
            return (pl.col(self.left) * pl.col(self.right)).list.sum()

        dot_product = (
            df.select(self.left, self.right)
            .with_row_index(name="index")
            .explode(self.left, self.right)
            .group_by("index", maintain_order=True)
            .agg(pl.col(self.left).dot(self.right).alias(alias))
            .drop("index")
        )
        return pl.concat([df, dot_product], how="horizontal")

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            transformation=ListDotProduct("left", "right"),
            input={
                "left": [[1, 2, 3], [2, 3]],
                "right": [[1, 1, 1], [2, 2]],
            },
            output=[6, 10],
        )


@dataclass
class HashColumns(Transformation, PolarsExprTransformation):
    columns: list[str]

    name = "hash_columns"
    dtype = FeatureType.uint64()

    def needed_columns(self) -> list[str]:
        return self.columns

    def polars_expr(self) -> pl.Expr:
        return pl.concat_str(self.columns).hash()

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        pl_df = pl.from_pandas(df)
        res = await self.transform_polars(pl_df.lazy(), "output", store)
        if isinstance(res, pl.Expr):
            return pl_df.with_columns(res.alias("output"))["output"].to_pandas()
        else:
            return res.collect()["output"].to_pandas()


@dataclass
class MultiTransformation(Transformation):
    transformations: list[tuple[Transformation, str | None]]
    name = "multi"
    dtype = FeatureType.string()

    def needed_columns(self) -> list[str]:
        all_col = []
        for tran, _ in self.transformations:
            all_col.extend(tran.needed_columns())
        return all_col

    async def transform_polars(
        self, df: pl.LazyFrame, alias: str, store: ContractStore
    ) -> pl.LazyFrame | pl.Expr:
        exclude_cols = []

        for tran, sub_alias in self.transformations:
            output = await tran.transform_polars(df, sub_alias or alias, store)

            if sub_alias:
                exclude_cols.append(sub_alias)

            if isinstance(output, pl.Expr):
                df = df.with_columns(output.alias(sub_alias or alias))
            else:
                df = output

        if alias in exclude_cols:
            exclude_cols.remove(alias)

        return df.select(pl.exclude(exclude_cols))

    async def transform_pandas(
        self, df: pd.DataFrame, store: ContractStore
    ) -> pd.Series:
        pl_df = pl.from_pandas(df)
        res = await self.transform_polars(pl_df.lazy(), "output", store)
        if isinstance(res, pl.Expr):
            return pl_df.with_columns(res.alias("output"))["output"].to_pandas()
        else:
            return res.collect()["output"].to_pandas()
