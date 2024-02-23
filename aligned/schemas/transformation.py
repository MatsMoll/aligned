from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import polars as pl
from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable
from aligned.schemas.feature import FeatureType
from aligned.schemas.literal_value import LiteralValue
from aligned.schemas.text_vectoriser import EmbeddingModel

if TYPE_CHECKING:
    from aligned.sources.s3 import AwsS3Config


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
        return pl.from_dict(self.input)

    @property
    def output_polars(self) -> pl.Series:
        try:
            values = pl.Series(self.output).fill_nan(None)
            if self.transformation.dtype == FeatureType.bool():
                return values.cast(pl.Boolean)
            else:
                return values
        except pl.InvalidOperationError:
            return pl.Series(self.output)


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


class Transformation(Codable, SerializableType):
    name: str
    dtype: FeatureType

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError(type(self))

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr | pl.Expr:
        raise NotImplementedError(type(self))

    def _serialize(self) -> dict:
        return self.to_dict()

    def should_skip(self, output_column: str, columns: list[str]) -> bool:
        return output_column in columns

    @classmethod
    def _deserialize(cls, value: dict) -> Transformation:
        name_type = value['name']
        del value['name']
        data_class = SupportedTransformations.shared().types[name_type]
        with suppress(AttributeError):
            if data_class.dtype:
                del value['dtype']

        return data_class.from_dict(value)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        raise NotImplementedError()

    @classmethod
    async def run_transformation_test_polars(cls) -> None:
        from polars.testing import assert_series_equal

        try:
            test = cls.test_definition()
            alias = 'something'
            output_df = await test.transformation.transform_polars(test.input_polars.lazy(), alias=alias)
            if isinstance(output_df, pl.Expr):
                output_df = test.input_polars.lazy().with_columns([output_df.alias(alias)])
            output = output_df.select(pl.col(alias)).collect().to_series()
            assert (set(test.input_polars.columns) - set(output_df.columns)) == set()

            expected = test.output_polars
            if test.transformation.dtype == FeatureType.bool():
                is_correct = output.equals(test.output_polars.alias(alias))
                assert is_correct, (
                    f'Output for {cls.__name__} is not correct.,'
                    f'\nGot: {output},\nexpected: {test.output_polars}'
                )
            else:
                assert_series_equal(expected.alias(alias), output, check_names=False, check_dtype=False)
        except AttributeError:
            raise AssertionError(
                f'Error for transformation {cls.__name__}. Could be missing a return in the transformation'
            )
        except NotImplementedError:
            pass

    @classmethod
    async def run_transformation_test_pandas(cls) -> None:
        import numpy as np
        from numpy.testing import assert_almost_equal

        with suppress(NotImplementedError):
            test = cls.test_definition()
            output = await test.transformation.transform_pandas(test.input_pandas)
            if test.transformation.dtype == FeatureType.bool():
                is_correct = np.all(output == test.output_pandas) | output.equals(test.output_pandas)
                assert is_correct, (
                    f'Output for {cls.__name__} is not correct.,'
                    f'\nGot: {output},\nexpected: {test.output_pandas}'
                )
            elif test.transformation.dtype == FeatureType.string():
                expected = test.output_pandas
                assert expected.equals(output), (
                    f'Output for {cls.__name__} is not correct.,'
                    f'\nGot: {output},\nexpected: {test.output_pandas}'
                )
            else:
                expected = test.output_pandas.to_numpy()
                output_np = output.to_numpy().astype('float')
                is_null = np.isnan(expected) & np.isnan(output_np)
                assert_almost_equal(expected[~is_null], output_np[~is_null])


class SupportedTransformations:

    types: dict[str, type[Transformation]]

    _shared: SupportedTransformations | None = None

    def __init__(self) -> None:
        self.types = {}

        for tran_type in [
            Equals,
            EqualsLiteral,
            NotEquals,
            NotEqualsLiteral,
            NotNull,
            PandasLambdaTransformation,
            PandasFunctionTransformation,
            PolarsLambdaTransformation,
            Ratio,
            StructField,
            DivideDenumeratorValue,
            Contains,
            GreaterThen,
            GreaterThenValue,
            GreaterThenOrEqual,
            LowerThen,
            LowerThenOrEqual,
            DateComponent,
            Subtraction,
            SubtractionValue,
            Multiply,
            MultiplyValue,
            Addition,
            AdditionValue,
            TimeDifference,
            Logarithm,
            LogarithmOnePluss,
            ToNumerical,
            ReplaceStrings,
            IsIn,
            And,
            Or,
            Inverse,
            Ordinal,
            FillNaValues,
            FillNaValuesColumns,
            Absolute,
            Round,
            Ceil,
            Floor,
            CopyTransformation,
            WordVectoriser,
            MapArgMax,
            LoadImageUrl,
            GrayscaleImage,
            Power,
            PowerFeature,
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
            ArrayAtIndex,
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
class PandasFunctionTransformation(Transformation):
    """
    This will encode a custom method, that is not a lambda function
    Threfore, we will stort the actuall code, and dynamically load it on runtime.

    This is unsafe, but will remove the ModuleImportError for custom methods
    """

    code: str
    function_name: str
    dtype: FeatureType
    name: str = 'pandas_code_tran'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        if self.function_name not in locals():
            exec(self.code)

        loaded = locals()[self.function_name]
        if asyncio.iscoroutinefunction(loaded):
            return await loaded(df)
        else:
            return loaded(df)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        pandas_df = df.collect().to_pandas()
        if self.function_name not in locals():
            exec(self.code)

        loaded = locals()[self.function_name]
        if asyncio.iscoroutinefunction(loaded):
            pandas_df[alias] = await loaded(pandas_df)
        else:
            pandas_df[alias] = loaded(pandas_df)

        return pl.from_pandas(pandas_df).lazy()

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            transformation=PandasFunctionTransformation(
                code='async def test(df):\n    return df["a"] + df["b"]',
                function_name='test',
                dtype=FeatureType.int32(),
            ),
            input={
                'a': [1, 2, 3, 4, 5],
                'b': [1, 2, 3, 4, 5],
            },
            output=[2, 4, 6, 8, 10],
        )


@dataclass
class PandasLambdaTransformation(Transformation):

    method: bytes
    code: str
    dtype: FeatureType
    name: str = 'pandas_lambda_tran'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        import asyncio

        import dill

        loaded = dill.loads(self.method)
        if asyncio.iscoroutinefunction(loaded):
            return await loaded(df)
        else:
            return loaded(df)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:

        import dill

        pandas_df = df.collect().to_pandas()
        loaded = dill.loads(self.method)
        if asyncio.iscoroutinefunction(loaded):
            pandas_df[alias] = await loaded(pandas_df)
        else:
            pandas_df[alias] = loaded(pandas_df)

        return pl.from_pandas(pandas_df).lazy()


@dataclass
class PolarsFunctionTransformation(Transformation):
    """
    This will encode a custom method, that is not a lambda function
    Threfore, we will stort the actuall code, and dynamically load it on runtime.

    This is unsafe, but will remove the ModuleImportError for custom methods
    """

    code: str
    function_name: str
    dtype: FeatureType
    name: str = 'pandas_code_tran'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        polars_df = await self.transform_polars(pl.from_pandas(df).lazy(), self.function_name)
        return polars_df.collect().to_pandas()[self.function_name]

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        if self.function_name not in locals():
            exec(self.code)

        loaded = locals()[self.function_name]
        if asyncio.iscoroutinefunction(loaded):
            return await loaded(df, alias)
        else:
            return loaded(df, alias)


@dataclass
class PolarsLambdaTransformation(Transformation):

    method: bytes
    code: str
    dtype: FeatureType
    name: str = 'polars_lambda_tran'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        import dill

        loaded: pl.Expr = dill.loads(self.method)
        pl_df = pl.from_pandas(df)
        pl_df = pl_df.with_columns((loaded).alias('polars_tran_column'))
        return pl_df['polars_tran_column'].to_pandas()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        import dill

        tran: Callable[[pl.LazyFrame, str], pl.LazyFrame] = dill.loads(self.method)
        if isinstance(tran, pl.Expr):
            return tran
        else:
            return tran(df, alias)


@dataclass
class NotNull(Transformation):

    key: str

    name: str = 'not_null'
    dtype: FeatureType = FeatureType.bool()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].notnull()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns(pl.col(self.key).is_not_null().alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            NotNull('x'),
            input={'x': ['Hello', None, None, 'test', None]},
            output=[True, False, False, True, False],
        )


@dataclass
class Equals(Transformation):

    key: str
    other_key: str

    name: str = 'equals_feature'
    dtype: FeatureType = FeatureType.bool()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] == df[self.other_key]

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key) == pl.col(self.other_key)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Equals('x', 'y'),
            input={
                'x': ['Hello', 'Test', 'nah', 'test', 'Test'],
                'y': ['hello', 'Test', 'other', 'no', 'Test'],
            },
            output=[False, True, False, False, True],
        )


@dataclass
class EqualsLiteral(Transformation):

    key: str
    value: LiteralValue

    name: str = 'equals'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, key: str, value: LiteralValue) -> None:
        self.key = key
        self.value = value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] == self.value.python_value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key) == self.value.python_value

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            EqualsLiteral('x', LiteralValue.from_value('Test')),
            input={'x': ['Hello', 'Test', 'nah', 'test', 'Test']},
            output=[False, True, False, False, True],
        )


@dataclass
class And(Transformation):

    first_key: str
    second_key: str

    name: str = 'and'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, first_key: str, second_key: str) -> None:
        self.first_key = first_key
        self.second_key = second_key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.first_key].isnull() | df[self.second_key].isnull()),
            transformation=lambda dfv: dfv[self.first_key] & dfv[self.second_key],
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns(
            (
                pl.when(pl.col(self.first_key).is_not_null() & pl.col(self.second_key).is_not_null())
                .then(pl.col(self.first_key) & pl.col(self.second_key))
                .otherwise(pl.lit(None))
            ).alias(alias)
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            And('x', 'y'),
            input={'x': [False, True, True, False, None], 'y': [True, False, True, False, False]},
            output=[False, False, True, False, np.nan],
        )


@dataclass
class Or(Transformation):

    first_key: str
    second_key: str

    name: str = 'or'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, first_key: str, second_key: str) -> None:
        self.first_key = first_key
        self.second_key = second_key

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns((pl.col(self.first_key) | pl.col(self.second_key)).alias(alias))

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        df[self.first_key].__invert__
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.first_key].isnull() | df[self.second_key].isnull()),
            transformation=lambda dfv: dfv[self.first_key] | dfv[self.second_key],
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Or('x', 'y'),
            input={'x': [False, True, True, False, None], 'y': [True, False, True, False, False]},
            output=[True, True, True, False, np.nan],
        )


@dataclass
class Inverse(Transformation):

    key: str

    name: str = 'inverse'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, key: str) -> None:
        self.key = key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isnull()),
            transformation=lambda dfv: ~dfv[self.key].astype('bool'),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns((~pl.col(self.key)).alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Inverse('x'),
            input={'x': [False, True, True, False, None]},
            output=[True, False, False, True, np.nan],
        )


@dataclass
class NotEquals(Transformation):

    key: str
    other_key: str

    name: str = 'not-equals-feature'
    dtype: FeatureType = FeatureType.bool()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] != df[self.other_key]

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key) != pl.col(self.other_key)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            NotEquals('x', 'y'),
            input={
                'x': ['Hello', 'Test', 'nah', 'test', 'Test'],
                'y': ['hello', 'Test', 'other', 'no', 'Test'],
            },
            output=[True, False, True, True, False],
        )


@dataclass
class NotEqualsLiteral(Transformation):

    key: str
    value: LiteralValue

    name: str = 'not-equals'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, key: str, value: Any) -> None:
        self.key = key
        if isinstance(value, LiteralValue):
            self.value = value
        else:
            self.value = LiteralValue.from_value(value)

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] != self.value.python_value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key) != self.value.python_value

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            NotEqualsLiteral('x', LiteralValue.from_value('Test')),
            input={'x': ['Hello', 'Test', 'nah', 'test', 'Test']},
            output=[True, False, True, True, False],
        )


@dataclass
class GreaterThenValue(Transformation):

    key: str
    value: float

    name: str = 'gt'
    dtype: FeatureType = FeatureType.bool()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] > self.value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key) > self.value

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            GreaterThenValue(key='x', value=2),
            input={'x': [1, 2, 3]},
            output=[False, False, True],
        )


@dataclass
class GreaterThen(Transformation):

    left_key: str
    right_key: str

    name: str = field(default='gtf')
    dtype: FeatureType = field(default=FeatureType.bool())

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.left_key] > df[self.right_key]

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.left_key) > pl.col(self.right_key)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            GreaterThen(left_key='x', right_key='y'),
            input={'x': [1, 2, 3, 5], 'y': [3, 2, 1, nan]},
            output=[False, False, True, False],
        )


@dataclass
class GreaterThenOrEqual(Transformation):

    key: str
    value: float

    name: str = 'gte'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] >= self.value,
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns((pl.col(self.key) >= self.value).alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            GreaterThenOrEqual(key='x', value=2),
            input={'x': [1, 2, 3, None]},
            output=[False, True, True, nan],
        )


@dataclass
class LowerThen(Transformation):

    key: str
    value: float

    name: str = 'lt'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] < self.value,
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns((pl.col(self.key) < self.value).alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            LowerThen(key='x', value=2), input={'x': [1, 2, 3, None]}, output=[True, False, False, nan]
        )


@dataclass
class LowerThenOrEqual(Transformation):

    key: str
    value: float

    name: str = 'lte'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key) <= self.value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] <= self.value,
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            LowerThenOrEqual(key='x', value=2), input={'x': [1, 2, 3, None]}, output=[True, True, False, nan]
        )


@dataclass
class SubtractionValue(Transformation, PsqlTransformation, RedshiftTransformation):

    front: str
    behind: LiteralValue

    name: str = 'sub_val'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, front: str, behind: LiteralValue) -> None:
        self.front = front
        self.behind = behind

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.front) - pl.lit(self.behind.python_value)

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.front].isna()),
            transformation=lambda dfv: dfv[self.front] - self.behind.python_value,
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            SubtractionValue(front='x', behind=LiteralValue.from_value(1)),
            input={'x': [1, 2, 0, None, 1]},
            output=[0, 1, -1, nan, 0],
        )

    def as_psql(self) -> str:
        return f'{self.front} - {self.behind.python_value}'


@dataclass
class Subtraction(Transformation, PsqlTransformation, RedshiftTransformation):

    front: str
    behind: str

    name: str = 'sub'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, front: str, behind: str) -> None:
        self.front = front
        self.behind = behind

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.front) - pl.col(self.behind)

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.front].isna() | df[self.behind].isna()),
            transformation=lambda dfv: dfv[self.front] - dfv[self.behind],
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            Subtraction(front='x', behind='y'),
            input={'x': [1, 2, 0, None, 1], 'y': [1, 0, 2, 1, None]},
            output=[0, 2, -2, nan, nan],
        )

    def as_psql(self) -> str:
        return f'{self.front} - {self.behind}'


@dataclass
class AdditionValue(Transformation):

    feature: str
    value: LiteralValue

    name: str = 'add_value'
    dtype: FeatureType = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.feature] + self.value.python_value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.feature) + pl.lit(self.value.python_value)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            AdditionValue(feature='x', value=LiteralValue.from_value(2)),
            input={'x': [1, 2, 0, None, 1], 'y': [1, 0, 2, 1, None]},
            output=[3, 4, 2, nan, 3],
        )


@dataclass
class Multiply(Transformation, PsqlTransformation, RedshiftTransformation):

    front: str
    behind: str

    name: str = 'mul'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, front: str, behind: str) -> None:
        self.front = front
        self.behind = behind

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.front] * df[self.behind]

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.front) * pl.col(self.behind)

    def as_psql(self) -> str:
        return f'{self.front} * {self.behind}'


@dataclass
class MultiplyValue(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str
    value: LiteralValue

    name: str = 'mul_val'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, key: str, value: LiteralValue) -> None:
        self.key = key
        self.value = value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key) * pl.lit(self.value.python_value)

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] * self.value.python_value

    def as_psql(self) -> str:
        return f"{self.key} * '{self.value.python_value}'"


@dataclass
class Addition(Transformation, PsqlTransformation, RedshiftTransformation):

    front: str
    behind: str

    name: str = 'add'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, front: str, behind: str) -> None:
        self.front = front
        self.behind = behind

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.front].isna() | df[self.behind].isna()),
            transformation=lambda dfv: dfv[self.front] + dfv[self.behind],
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.front) + pl.col(self.behind)

    def as_psql(self) -> str:
        return f'{self.front} + {self.behind}'

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            Addition(front='x', behind='y'),
            input={'x': [1, 2, 0, None, 1], 'y': [1, 0, 2, 1, None]},
            output=[2, 2, 2, nan, nan],
        )


@dataclass
class TimeDifference(Transformation, PsqlTransformation, RedshiftTransformation):

    front: str
    behind: str
    unit: str

    name: str = 'time-diff'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, front: str, behind: str, unit: str = 's') -> None:
        self.front = front
        self.behind = behind
        self.unit = unit

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.front].isna() | df[self.behind].isna()),
            transformation=lambda dfv: (dfv[self.front] - dfv[self.behind]) / np.timedelta64(1, self.unit),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns((pl.col(self.front) - pl.col(self.behind)).dt.seconds().alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            TimeDifference(front='x', behind='y'),
            input={
                'x': [
                    datetime.fromtimestamp(1),
                    datetime.fromtimestamp(2),
                    datetime.fromtimestamp(0),
                    None,
                    datetime.fromtimestamp(1),
                ],
                'y': [
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
class Logarithm(Transformation):

    key: str

    name: str = 'log'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, key: str) -> None:
        self.key = key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | (df[self.key] <= 0)),
            transformation=lambda dfv: np.log(dfv[self.key]),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns(
            (pl.when(pl.col(self.key) > 0).then(pl.col(self.key).log()).otherwise(pl.lit(None))).alias(alias)
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            Logarithm('x'), input={'x': [1, 0, np.e, None, -1]}, output=[0, nan, 1, nan, nan]
        )


@dataclass
class LogarithmOnePluss(Transformation):

    key: str

    name: str = 'log1p'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, key: str) -> None:
        self.key = key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | (df[self.key] <= -1)),
            transformation=lambda dfv: np.log1p(dfv[self.key]),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns(
            (pl.when(pl.col(self.key) > -1).then((pl.col(self.key) + 1).log()).otherwise(pl.lit(None))).alias(
                alias
            )
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            LogarithmOnePluss('x'),
            input={'x': [1, 0, np.e - 1, None, -1]},
            output=[0.6931471806, 0, 1, nan, nan],
        )


@dataclass
class ToNumerical(Transformation):

    key: str

    name: str = 'to-num'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, key: str) -> None:
        self.key = key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from pandas import to_numeric

        return to_numeric(df[self.key], errors='coerce')

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).cast(pl.Float64)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            ToNumerical('x'),
            input={'x': ['1', '0', '10.5', None, nan, '-20']},
            output=[1, 0, 10.5, nan, nan, -20],
        )


@dataclass
class DateComponent(Transformation):

    key: str
    component: str

    name: str = 'date-component'
    dtype: FeatureType = FeatureType.int32()

    def __init__(self, key: str, component: str) -> None:
        self.key = key
        self.component = component

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:

        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna()),
            transformation=lambda dfv: getattr(dfv[self.key].dt, self.component),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        col = pl.col(self.key).cast(pl.Datetime).dt
        match self.component:
            case 'day':
                expr = col.day()
            case 'days':
                expr = col.days()
            case 'epoch':
                expr = col.epoch()
            case 'hour':
                expr = col.hour()
            case 'hours':
                expr = col.hours()
            case 'iso_year':
                expr = col.iso_year()
            case 'microsecond':
                expr = col.microsecond()
            case 'microseconds':
                expr = col.microseconds()
            case 'millisecond':
                expr = col.millisecond()
            case 'milliseconds':
                expr = col.milliseconds()
            case 'minute':
                expr = col.minute()
            case 'minutes':
                expr = col.minutes()
            case 'month':
                expr = col.month()
            case 'nanosecond':
                expr = col.nanosecond()
            case 'nanoseconds':
                expr = col.nanoseconds()
            case 'ordinal_day':
                expr = col.ordinal_day()
            case 'quarter':
                expr = col.quarter()
            case 'second':
                expr = col.second()
            case 'seconds':
                expr = col.seconds()
            case 'week':
                expr = col.week()
            case 'weekday':
                expr = col.weekday()
            case 'year':
                expr = col.year()
            case 'dayofweek':
                expr = col.weekday()
            case _:
                raise NotImplementedError(
                    f'Date component {self.component} is not implemented. Maybe setup a PR and contribute?'
                )
        return expr

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            DateComponent(key='x', component='hour'),
            input={
                'x': [
                    datetime.fromisoformat(value) if value else None
                    for value in ['2022-04-02T20:20:50', None, '2022-02-20T23:20:50', '1993-04-02T01:20:50']
                ]
            },
            output=[20, nan, 23, 1],
        )


@dataclass
class ArrayAtIndex(Transformation):
    """Checks if an array contains a value

    some_array = List(String())
    contains_a_char = some_array.contains("a")
    """

    key: str
    index: int

    name: str = 'array_at_index'
    dtype: FeatureType = FeatureType.bool()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return pl.Series(df[self.key]).list.get(self.index).to_pandas()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).list.get(self.index).alias(alias)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            ArrayContains('x', LiteralValue.from_value('test')),
            input={'x': [['Hello', 'test'], ['nah'], ['test', 'espania', None]]},
            output=[True, False, True],
        )


@dataclass
class ArrayContains(Transformation):
    """Checks if an array contains a value

    some_array = List(String())
    contains_a_char = some_array.contains("a")
    """

    key: str
    value: LiteralValue

    name: str = 'array_contains'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, key: str, value: Any | LiteralValue) -> None:
        self.key = key
        if isinstance(value, LiteralValue):
            self.value = value
        else:
            self.value = LiteralValue.from_value(value)

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return pl.Series(df[self.key]).list.contains(self.value.python_value).to_pandas()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).list.contains(self.value.python_value)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            ArrayContains('x', LiteralValue.from_value('test')),
            input={'x': [['Hello', 'test'], ['nah'], ['test', 'espania', None]]},
            output=[True, False, True],
        )


@dataclass
class Contains(Transformation):
    """Checks if a string value contains another string

    some_string = String()
    contains_a_char = some_string.contains("a")
    """

    key: str
    value: str

    name: str = 'contains'
    dtype: FeatureType = FeatureType.bool()

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna()),
            transformation=lambda dfv: dfv[self.key].astype('str').str.contains(self.value),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).str.contains(self.value)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Contains('x', 'es'),
            input={'x': ['Hello', 'Test', 'nah', 'test', 'espania', None]},
            output=[False, True, False, True, True, None],
        )


@dataclass
class Ordinal(Transformation):

    key: str
    orders: list[str]

    @property
    def orders_dict(self) -> dict[str, int]:
        return {key: index for index, key in enumerate(self.orders)}

    name: str = 'ordinal'
    dtype: FeatureType = FeatureType.int32()

    def __init__(self, key: str, orders: list[str]) -> None:
        self.key = key
        self.orders = orders

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].map(self.orders_dict)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        mapper = pl.DataFrame({self.key: list(self.orders), alias: list(range(0, len(self.orders)))})
        return df.join(mapper.lazy(), on=self.key, how='left')

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            Ordinal('x', ['a', 'b', 'c', 'd']),
            input={'x': ['a', 'b', 'a', None, 'd', 'p']},
            output=[0, 1, 0, nan, 3, nan],
        )


@dataclass
class ReplaceStrings(Transformation):

    key: str
    values: list[tuple[str, str]]

    name: str = 'replace'
    dtype: FeatureType = FeatureType.string()

    def __init__(self, key: str, values: list[tuple[str, str]]) -> None:
        self.key = key
        self.values = values

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        temp_df = df[self.key].copy()
        mask = ~(df[self.key].isna() | df[self.key].isnull())
        temp_df.loc[~mask] = np.nan
        for k, v in self.values:
            temp_df.loc[mask] = temp_df.loc[mask].str.replace(k, v, regex=True)

        return temp_df

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        collected = df.collect()
        pandas_column = collected.select(self.key).to_pandas()
        transformed = await self.transform_pandas(pandas_column)
        return collected.with_columns(pl.Series(transformed).alias(alias)).lazy()

    # @staticmethod
    # def test_definition() -> TransformationTestDefinition:
    #     from numpy import nan
    #
    #     return TransformationTestDefinition(
    #         ReplaceStrings('x', [
    #             (r'20[\s]*-[\s]*10', '15'),
    #             (' ', ''),
    #             ('.', ''),
    #             ('10-20', '15'),
    #             ('20\\+', '30')
    #         ]),
    #         input={'x': [' 20', '10 - 20', '.yeah', '20+', None, '20   - 10']},
    #         output=['20', '15', 'yeah', '30', nan, '15'],
    #     )


@dataclass
class Ratio(Transformation):

    numerator: str
    denumerator: str

    name: str = 'ratio'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, numerator: str, denumerator: str) -> None:
        self.numerator = numerator
        self.denumerator = denumerator

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(
                df[self.numerator].isna() | df[self.denumerator].isna() | df[self.denumerator] == 0
            ),
            transformation=lambda dfv: dfv[self.numerator].astype(float)
            / dfv[self.denumerator].astype(float),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return (
            pl.when(pl.col(self.denumerator) != 0)
            .then(pl.col(self.numerator) / pl.col(self.denumerator))
            .otherwise(pl.lit(None))
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            Ratio('x', 'y'),
            input={'x': [1, 2, 0, 1, None, 9], 'y': [1, 0, 1, 4, 2, None]},
            output=[1, nan, 0, 0.25, nan, nan],
        )


@dataclass
class DivideDenumeratorValue(Transformation):

    numerator: str
    denumerator: LiteralValue

    name: str = 'div_denum_val'
    dtype: FeatureType = FeatureType.float()

    def __init__(self, numerator: str, denumerator: LiteralValue) -> None:
        self.numerator = numerator
        self.denumerator = denumerator
        assert denumerator.python_value != 0

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.numerator].isna()),
            transformation=lambda dfv: dfv[self.numerator].astype(float) / self.denumerator.python_value,
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.numerator) / pl.lit(self.denumerator.python_value)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            DivideDenumeratorValue('x', LiteralValue.from_value(2)),
            input={'x': [1, 2, 0, 1, None, 9]},
            output=[0.5, 1, 0, 0.5, nan, 4.5],
        )


@dataclass
class IsIn(Transformation):

    values: list
    key: str

    name = 'isin'
    dtype = FeatureType.bool()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].isin(self.values)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).is_in(self.values)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            IsIn(values=['hello', 'test'], key='x'),
            input={'x': ['No', 'Hello', 'hello', 'test', 'nah', 'nehtest']},
            output=[False, False, True, True, False, False],
        )


@dataclass
class FillNaValuesColumns(Transformation):

    key: str
    fill_key: str
    dtype: FeatureType

    name: str = 'fill_missing_key'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].fillna(df[self.fill_key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        if self.dtype == FeatureType.float():
            return pl.col(self.key).fill_nan(pl.col(self.fill_key)).fill_null(pl.col(self.fill_key))

        else:
            return pl.col(self.key).fill_null(pl.col(self.fill_key))

    def should_skip(self, output_column: str, columns: list[str]) -> bool:
        return False

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            FillNaValuesColumns('x', 'y', dtype=FeatureType.int32()),
            input={'x': [1, 1, None, None, 3, 3, None, 4, 5, None], 'y': [1, 2, 1, 2, 7, 2, 4, 1, 1, 9]},
            output=[1, 1, 1, 2, 3, 3, 4, 4, 5, 9],
        )


@dataclass
class FillNaValues(Transformation):

    key: str
    value: LiteralValue
    dtype: FeatureType

    name: str = 'fill_missing'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].fillna(self.value.python_value)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        if self.dtype == FeatureType.float():
            return pl.col(self.key).fill_nan(self.value.python_value).fill_null(self.value.python_value)

        else:
            return pl.col(self.key).fill_null(self.value.python_value)

    def should_skip(self, output_column: str, columns: list[str]) -> bool:
        return False

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            FillNaValues('x', LiteralValue.from_value(3), dtype=FeatureType.int32()),
            input={'x': [1, 1, None, None, 3, 3, None, 4, 5, None]},
            output=[1, 1, 3, 3, 3, 3, 3, 4, 5, 3],
        )


@dataclass
class CopyTransformation(Transformation):
    key: str
    dtype: FeatureType

    name: str = 'nothing'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key]

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).alias(alias)


@dataclass
class Floor(Transformation):

    key: str
    dtype: FeatureType = FeatureType.int64()

    name: str = 'floor'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from numpy import floor

        return floor(df[self.key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).floor().alias(alias)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Floor('x'),
            input={'x': [1.3, 1.9, None]},
            output=[1, 1, None],
        )


@dataclass
class Ceil(Transformation):

    key: str
    dtype: FeatureType = FeatureType.int64()

    name: str = 'ceil'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from numpy import ceil

        return ceil(df[self.key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).ceil().alias(alias)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Ceil('x'),
            input={'x': [1.3, 1.9, None]},
            output=[2, 2, None],
        )


@dataclass
class Round(Transformation):

    key: str
    dtype: FeatureType = FeatureType.int64()

    name: str = 'round'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from numpy import round

        return round(df[self.key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).round(0).alias(alias)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Round('x'),
            input={'x': [1.3, 1.9, None]},
            output=[1, 2, None],
        )


@dataclass
class Absolute(Transformation):

    key: str
    dtype: FeatureType = FeatureType.float()

    name: str = 'abs'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from numpy import abs

        return abs(df[self.key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).abs().alias(alias)

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Absolute('x'),
            input={'x': [-13, 19, None]},
            output=[13, 19, None],
        )


@dataclass
class MapArgMax(Transformation):

    column_mappings: dict[str, LiteralValue]
    name = 'map_arg_max'

    @property
    def dtype(self) -> FeatureType:
        return list(self.column_mappings.values())[0].dtype

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        pl_df = await self.transform_polars(pl.from_pandas(df).lazy(), 'feature')
        return pl_df.collect().to_pandas()['feature']

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        expr: pl.Expr = pl.lit(None)

        if len(self.column_mappings) == 1:
            key, value = list(self.column_mappings.items())[0]
            if self.dtype == FeatureType.bool():
                expr = pl.when(pl.col(key) > 0.5).then(value.python_value).otherwise(not value.python_value)
            elif self.dtype == FeatureType.string():
                expr = (
                    pl.when(pl.col(key) > 0.5).then(value.python_value).otherwise(f'not {value.python_value}')
                )
            else:
                expr = pl.when(pl.col(key) > 0.5).then(value.python_value).otherwise(pl.lit(None))
            return df.with_columns(expr.alias(alias))
        else:
            features = list(self.column_mappings.keys())
            arg_max_alias = f'{alias}_arg_max'
            array_row_alias = f'{alias}_row'
            mapper = pl.DataFrame(
                {
                    alias: [self.column_mappings[feature].python_value for feature in features],
                    arg_max_alias: list(range(0, len(features))),
                }
            ).with_columns(pl.col(arg_max_alias).cast(pl.UInt32))
            sub = df.with_columns(pl.concat_list(pl.col(features)).alias(array_row_alias)).with_columns(
                pl.col(array_row_alias).list.arg_max().alias(arg_max_alias)
            )
            return sub.join(mapper.lazy(), on=arg_max_alias, how='left').select(
                pl.exclude([arg_max_alias, array_row_alias])
            )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            MapArgMax(
                {
                    'a_prob': LiteralValue.from_value('a'),
                    'b_prob': LiteralValue.from_value('b'),
                    'c_prob': LiteralValue.from_value('c'),
                }
            ),
            input={'a_prob': [0.01, 0.9, 0.25], 'b_prob': [0.9, 0.05, 0.15], 'c_prob': [0.09, 0.05, 0.6]},
            output=['b', 'a', 'c'],
        )


@dataclass
class WordVectoriser(Transformation):
    key: str
    model: EmbeddingModel

    name = 'word_vectoriser'
    dtype = FeatureType.embedding()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return await self.model.vectorise_pandas(df[self.key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return await self.model.vectorise_polars(df, self.key, alias)


@dataclass
class LoadImageUrl(Transformation):

    image_url_key: str

    name = 'load_image'
    dtype = FeatureType.array()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        import asyncio
        from io import BytesIO

        import numpy as np
        from PIL import Image

        from aligned.sources.local import StorageFileSource

        urls = df.select(self.image_url_key).collect()[self.image_url_key]

        images = await asyncio.gather(*[StorageFileSource(url).read() for url in urls.to_list()])
        data = [np.asarray(Image.open(BytesIO(buffer))) for buffer in images]
        image_dfs = pl.DataFrame({alias: data})
        return df.with_context(image_dfs.lazy()).select(pl.all())


@dataclass
class GrayscaleImage(Transformation):

    image_key: str

    name = 'grayscale_image'
    dtype = FeatureType.array()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        import numpy as np

        def grayscale(images):
            return pl.Series(
                [np.mean(image, axis=2) if len(image.shape) == 3 else image for image in images.to_list()]
            )

        return pl.col(self.image_key).map(grayscale).alias(alias)


@dataclass
class Power(Transformation):

    key: str
    power: LiteralValue
    name = 'power'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] ** self.power.python_value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).pow(self.power.python_value)


@dataclass
class PowerFeature(Transformation):

    key: str
    power_key: float
    name = 'power_feat'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] ** df[self.power_key]

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).pow(pl.col(self.power_key))


@dataclass
class AppendConstString(Transformation):

    key: str
    string: str

    name = 'append_const_string'
    dtype = FeatureType.string()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] + self.string

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.concat_str([pl.col(self.key).fill_null(''), pl.lit(self.string)], separator='').alias(alias)


@dataclass
class AppendStrings(Transformation):

    first_key: str
    second_key: str
    sep: str

    name = 'append_strings'
    dtype = FeatureType.string()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.first_key] + self.sep + df[self.second_key]

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns(
            pl.concat_str(
                [pl.col(self.first_key).fill_null(''), pl.col(self.second_key).fill_null('')],
                separator=self.sep,
            ).alias(alias)
        )


@dataclass
class PrependConstString(Transformation):

    string: str
    key: str

    name = 'prepend_const_string'
    dtype = FeatureType.string()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return self.string + df[self.key]

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.concat_str([pl.lit(self.string), pl.col(self.key).fill_null('')], separator='').alias(alias)


@dataclass
class ConcatStringAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str
    separator: str = field(default=' ')

    name = 'concat_string_agg'
    dtype = FeatureType.string()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return (
            (await self.transform_polars(pl.from_pandas(df).lazy(), self.name))
            .collect()
            .to_pandas()[self.name]
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return df.with_columns(pl.concat_str(pl.col(self.key), sep=self.separator).alias(alias))

    def as_psql(self) -> str:
        return f"array_to_string(array_agg({self.key}), '{self.separator}')"

    def as_redshift(self) -> str:
        return f"listagg(\"{self.key}\", '{self.separator}') within group (order by \"{self.key}\")"


@dataclass
class SumAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str

    name = 'sum_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.sum(self.key)

    def as_psql(self) -> str:
        return f'SUM({self.key})'


@dataclass
class MeanAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str

    name = 'mean_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).mean()

    def as_psql(self) -> str:
        return f'AVG({self.key})'


@dataclass
class MinAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str

    name = 'min_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).min()

    def as_psql(self) -> str:
        return f'MIN({self.key})'


@dataclass
class MaxAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str

    name = 'max_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).max()

    def as_psql(self) -> str:
        return f'MAX({self.key})'


@dataclass
class CountAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str

    name = 'count_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).count()

    def as_psql(self) -> str:
        return f'COUNT({self.key})'


@dataclass
class CountDistinctAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str

    name = 'count_distinct_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).unique_counts()

    def as_psql(self) -> str:
        return f'COUNT(DISTINCT {self.key})'


@dataclass
class StdAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str

    name = 'std_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).std()

    def as_psql(self) -> str:
        return f'STDDEV({self.key})'


@dataclass
class VarianceAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str

    name = 'var_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).var()

    def as_psql(self) -> str:
        return f'variance({self.key})'


@dataclass
class MedianAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str

    name = 'median_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).median()

    def as_psql(self) -> str:
        return f'percentile_cont(0.5) WITHIN GROUP(ORDER BY {self.key})'


@dataclass
class PercentileAggregation(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str
    percentile: float

    name = 'percentile_agg'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).quantile(self.percentile)

    def as_psql(self) -> str:
        return f'percentile_cont({self.percentile}) WITHIN GROUP(ORDER BY {self.key})'


@dataclass
class Clip(Transformation, PsqlTransformation, RedshiftTransformation):

    key: str
    lower: LiteralValue
    upper: LiteralValue

    name = 'clip'
    dtype = FeatureType.float()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].clip(lower=self.lower.python_value, upper=self.upper.python_value)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).clip(lower_bound=self.lower.python_value, upper_bound=self.upper.python_value)

    def as_psql(self) -> str:
        return (
            f'CASE WHEN {self.key} < {self.lower} THEN {self.lower} WHEN '
            f'{self.key} > {self.upper} THEN {self.upper} ELSE {self.key} END'
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            transformation=Clip(key='a', lower=LiteralValue.from_value(0), upper=LiteralValue.from_value(1)),
            input={'a': [-1, 0.1, 0.9, 2]},
            output=[0, 0.1, 0.9, 1],
        )


@dataclass
class PresignedAwsUrl(Transformation):

    config: AwsS3Config
    key: str

    max_age_seconds: int = field(default=30)

    name = 'presigned_aws_url'
    dtype = FeatureType.string()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from aioaws.s3 import S3Client
        from httpx import AsyncClient

        s3 = S3Client(AsyncClient(), config=self.config.s3_config)
        return df[self.key].apply(lambda x: s3.signed_download_url(x, max_age=self.max_age_seconds))

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        from aioaws.s3 import S3Client
        from httpx import AsyncClient

        s3 = S3Client(AsyncClient(), config=self.config.s3_config)

        return df.with_columns(
            pl.col(self.key)
            .apply(lambda x: s3.signed_download_url(x, max_age=self.max_age_seconds))
            .alias(alias)
        )


@dataclass
class StructField(Transformation):

    key: str
    field: str

    name = 'struct_field'
    dtype = FeatureType.string()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        data = pl.from_pandas(df).lazy()
        tran = await self.transform_polars(data, 'feature')

        if isinstance(tran, pl.LazyFrame):
            return tran.collect().to_pandas()['feature']

        return data.select(tran).collect().to_pandas()['feature']

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        if df.schema[self.key].is_(pl.Utf8):
            return await JsonPath(self.key, f'$.{self.field}').transform_polars(df, alias)
        else:
            return pl.col(self.key).struct.field(self.field).alias(alias)


@dataclass
class JsonPath(Transformation):

    key: str
    path: str

    name = 'json_path'
    dtype = FeatureType.string()

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return pl.Series(df[self.key]).str.json_path_match(self.path).to_pandas()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr:
        return pl.col(self.key).str.json_path_match(self.path).alias(alias)


@dataclass
class Split(Transformation):

    key: str
    separator: str

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].str.split(self.separator)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame | pl.Expr | pl.Expr:
        return pl.col(self.key).str.split(self.separator)
