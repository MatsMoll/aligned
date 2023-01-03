from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import polars as pl
from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable
from aligned.schemas.feature import FeatureType


@dataclass
class TransformationTestDefinition:
    transformation: 'Transformation'
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
            if self.transformation.dtype == FeatureType('').bool:
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
    result.loc[is_valid_mask] = transformation(df.loc[is_valid_mask])
    return result


class Transformation(Codable, SerializableType):
    name: str
    dtype: FeatureType

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        pass

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        raise NotImplementedError()

    def _serialize(self) -> dict:
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> 'Transformation':
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
            output_df = (
                await test.transformation.transform_polars(test.input_polars.lazy(), alias=alias)
            ).collect()
            output = output_df.select(pl.col(alias)).to_series()
            assert (set(test.input_polars.columns) - set(output_df.columns)) == set()

            expected = test.output_polars
            if test.transformation.dtype == FeatureType('').bool:
                is_correct = output.series_equal(test.output_polars.alias(alias))
                assert is_correct, (
                    f'Output for {cls.__name__} is not correct.,'
                    f'\nGot: {output},\nexpected: {test.output_polars}'
                )
            else:
                assert_series_equal(expected.alias(alias), output, check_names=False, check_dtype=False)
        except pl.NotFoundError:
            AssertionError(
                f'Not able to find resulting transformation {cls.__name__}, remember to add .alias(alias)'
            )
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
            if test.transformation.dtype == FeatureType('').bool:
                is_correct = np.all(output == test.output_pandas) | output.equals(test.output_pandas)
                assert is_correct, (
                    f'Output for {cls.__name__} is not correct.,'
                    f'\nGot: {output},\nexpected: {test.output_pandas}'
                )
            elif test.transformation.dtype == FeatureType('').string:
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

    _shared: Optional['SupportedTransformations'] = None

    def __init__(self) -> None:
        self.types = {}

        for tran_type in [
            Equals,
            NotEquals,
            DillTransformation,
            StandardScalingTransformation,
            Ratio,
            Contains,
            GreaterThen,
            GreaterThenValue,
            GreaterThenOrEqual,
            LowerThen,
            LowerThenOrEqual,
            DateComponent,
            Subtraction,
            Addition,
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
            Absolute,
            Round,
            Ceil,
            Floor,
            CopyTransformation,
        ]:
            self.add(tran_type)

    def add(self, transformation: type[Transformation]) -> None:
        self.types[transformation.name] = transformation

    @classmethod
    def shared(cls) -> 'SupportedTransformations':
        if cls._shared:
            return cls._shared
        cls._shared = SupportedTransformations()
        return cls._shared


@dataclass
class DillTransformation(Transformation):

    method: bytes
    dtype: FeatureType
    name: str = 'custom_transformation'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        import dill

        loaded = dill.loads(self.method)
        return await loaded(df)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return await super().transform_polars()


@dataclass
class Equals(Transformation):

    key: str
    value: str

    name: str = 'equals'
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] == self.value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.key) == self.value).alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Equals('x', 'Test'),
            input={'x': ['Hello', 'Test', 'nah', 'test', 'Test']},
            output=[False, True, False, False, True],
        )


@dataclass
class And(Transformation):

    first_key: str
    second_key: str

    name: str = 'and'
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, first_key: str, second_key: str) -> None:
        self.first_key = first_key
        self.second_key = second_key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.first_key].isnull() | df[self.second_key].isnull()),
            transformation=lambda dfv: dfv[self.first_key] & dfv[self.second_key],
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(
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
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, first_key: str, second_key: str) -> None:
        self.first_key = first_key
        self.second_key = second_key

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.first_key) | pl.col(self.second_key)).alias(alias))

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
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str) -> None:
        self.key = key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] == False,  # noqa: E712
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((~pl.col(self.key)).alias(alias))

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
    value: str

    name: str = 'not-equals'
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key] != self.value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.key) != self.value).alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            NotEquals('x', 'Test'),
            input={'x': ['Hello', 'Test', 'nah', 'test', 'Test']},
            output=[True, False, True, True, False],
        )


@dataclass
class GreaterThenValue(Transformation):

    key: str
    value: float

    name: str = 'gt'
    dtype: FeatureType = FeatureType('').bool

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] > self.value,
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(
            (
                pl.when(pl.col(self.key).is_not_null() & pl.col(self.key).is_not_nan())
                .then(pl.col(self.key) > self.value)
                .otherwise(pl.lit(None))
            ).alias(alias)
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            GreaterThenValue(key='x', value=2), input={'x': [1, 2, 3, nan]}, output=[False, False, True, nan]
        )


@dataclass
class GreaterThen(Transformation):

    left_key: str
    right_key: str

    name: str = field(default='gtf')
    dtype: FeatureType = field(default=FeatureType('').bool)

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(
                df[self.left_key].isna()
                | df[self.left_key].isnull()
                | df[self.right_key].isna()
                | df[self.right_key].isnull()
            ),
            transformation=lambda dfv: dfv[self.left_key] > dfv[self.right_key],
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(
            (
                pl.when(
                    pl.col(self.left_key).is_not_null()
                    & pl.col(self.right_key).is_not_null()
                    & pl.col(self.left_key).is_not_nan()
                    & pl.col(self.right_key).is_not_nan()
                )
                .then(pl.col(self.left_key) > pl.col(self.right_key))
                .otherwise(pl.lit(None))
            ).alias(alias)
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            GreaterThen(left_key='x', right_key='y'),
            input={'x': [1, 2, 3, nan, 5], 'y': [3, 2, 1, 5, nan]},
            output=[False, False, True, nan, nan],
        )


@dataclass
class GreaterThenOrEqual(Transformation):

    key: str
    value: float

    name: str = 'gte'
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] >= self.value,
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.key) >= self.value).alias(alias))

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
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] < self.value,
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.key) < self.value).alias(alias))

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
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.key) <= self.value).alias(alias))

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
class Subtraction(Transformation):

    front: str
    behind: str

    name: str = 'sub'
    dtype: FeatureType = FeatureType('').float

    def __init__(self, front: str, behind: str) -> None:
        self.front = front
        self.behind = behind

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.front) - pl.col(self.behind)).alias(alias))

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


@dataclass
class Addition(Transformation):

    front: str
    behind: str

    name: str = 'add'
    dtype: FeatureType = FeatureType('').float

    def __init__(self, front: str, behind: str) -> None:
        self.front = front
        self.behind = behind

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.front].isna() | df[self.behind].isna()),
            transformation=lambda dfv: dfv[self.front] + dfv[self.behind],
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.front) + pl.col(self.behind)).alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            Addition(front='x', behind='y'),
            input={'x': [1, 2, 0, None, 1], 'y': [1, 0, 2, 1, None]},
            output=[2, 2, 2, nan, nan],
        )


@dataclass
class TimeDifference(Transformation):

    front: str
    behind: str
    unit: str

    name: str = 'time-diff'
    dtype: FeatureType = FeatureType('').float

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

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.front) - pl.col(self.behind)).dt.seconds().alias(alias))

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


@dataclass
class Logarithm(Transformation):

    key: str

    name: str = 'log'
    dtype: FeatureType = FeatureType('').float

    def __init__(self, key: str) -> None:
        self.key = key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | (df[self.key] <= 0)),
            transformation=lambda dfv: np.log(dfv[self.key]),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(
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
    dtype: FeatureType = FeatureType('').float

    def __init__(self, key: str) -> None:
        self.key = key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | (df[self.key] <= -1)),
            transformation=lambda dfv: np.log1p(dfv[self.key]),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(
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
    dtype: FeatureType = FeatureType('').float

    def __init__(self, key: str) -> None:
        self.key = key

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from pandas import to_numeric

        return to_numeric(df[self.key], errors='coerce')

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column((pl.col(self.key).cast(pl.Float64)).alias(alias))

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
    dtype: FeatureType = FeatureType('').int32

    def __init__(self, key: str, component: str) -> None:
        self.key = key
        self.component = component

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:

        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna()),
            transformation=lambda dfv: getattr(dfv[self.key].dt, self.component),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        pl.col(self.key).str.strptime(pl.Datetime, strict=False)
        pl.when(pl.col(self.key))

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
            case _:
                raise NotImplementedError(
                    f'Date component {self.component} is not implemented. Maybe setup a PR and contribute?'
                )
        return df.with_column(expr.alias(alias))

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
class Contains(Transformation):
    """Checks if a string value contains another string

    some_string = String()
    contains_a_char = some_string.contains("a")
    """

    key: str
    value: str

    name: str = 'contains'
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna()),
            transformation=lambda dfv: dfv[self.key].astype('str').str.contains(self.value),
        )

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(pl.col(self.key).str.contains(self.value).alias(alias))

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
    dtype: FeatureType = FeatureType('').int32

    def __init__(self, key: str, orders: list[str]) -> None:
        self.key = key
        self.orders = orders

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].map(self.orders_dict)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
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
    values: dict[str, str]

    name: str = 'replace'
    dtype: FeatureType = FeatureType('').string

    def __init__(self, key: str, values: dict[str, str]) -> None:
        self.key = key
        self.values = values

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        temp_df = df[self.key].copy()
        mask = ~(df[self.key].isna() | df[self.key].isnull())
        temp_df.loc[~mask] = np.nan
        for k, v in self.values.items():
            temp_df.loc[mask] = temp_df.loc[mask].astype(str).str.replace(k, v)

        return temp_df

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        raise NotImplementedError()

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            ReplaceStrings('x', {r'20[\s]*-[\s]*10': '15', ' ': '', '.': '', '10-20': '15', '20\\+': '30'}),
            input={'x': [' 20', '10 - 20', '.yeah', '20+', None, '20   - 10']},
            output=['20', '15', 'yeah', '30', nan, '15'],
        )


@dataclass
class Ratio(Transformation):

    numerator: str
    denumerator: str

    name: str = 'ratio'
    dtype: FeatureType = FeatureType('').float

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

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(
            (
                pl.when(pl.col(self.denumerator) != 0)
                .then(pl.col(self.numerator) / pl.col(self.denumerator))
                .otherwise(pl.lit(None))
            ).alias(alias)
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
class StandardScalingTransformation(Transformation):

    mean: float
    std: float
    key: str

    name = 'standard_scaling'
    dtype = FeatureType('').float

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return (df[self.key] - self.mean) / self.std

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(((pl.col(self.key) - self.mean) / self.std).alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            StandardScalingTransformation(mean=1, std=0.5, key='x'),
            input={'x': [1, 1.5, 0.5, 1, 2, 3]},
            output=[0, 1, -1, 0, 2, 4],
        )


@dataclass
class IsIn(Transformation):

    values: list
    key: str

    name = 'isin'
    dtype = FeatureType('').bool

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].isin(self.values)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(pl.col(self.key).is_in(self.values).alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            IsIn(values=['hello', 'test'], key='x'),
            input={'x': ['No', 'Hello', 'hello', 'test', 'nah', 'nehtest']},
            output=[False, False, True, True, False, False],
        )


@dataclass
class FillNaValues(Transformation):

    key: str
    value: Any
    dtype: FeatureType

    name: str = 'fill_missing'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        return df[self.key].fillna(self.value)

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        if self.dtype == FeatureType('').float:
            return df.with_column(pl.col(self.key).fill_nan(self.value).fill_null(self.value).alias(alias))
        else:
            return df.with_column(pl.col(self.key).fill_null(self.value).alias(alias))

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            FillNaValues('x', 3, dtype=FeatureType('').int32),
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

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(pl.col(self.key).alias(alias))


@dataclass
class Floor(Transformation):

    key: str
    dtype: FeatureType = FeatureType('').int64

    name: str = 'floor'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from numpy import floor

        return floor(df[self.key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(pl.col(self.key).floor().alias(alias))


@dataclass
class Ceil(Transformation):

    key: str
    dtype: FeatureType = FeatureType('').int64

    name: str = 'ceil'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from numpy import ceil

        return ceil(df[self.key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(pl.col(self.key).ceil().alias(alias))


@dataclass
class Round(Transformation):

    key: str
    dtype: FeatureType = FeatureType('').int64

    name: str = 'round'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from numpy import round

        return round(df[self.key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(pl.col(self.key).round().alias(alias))


@dataclass
class Absolute(Transformation):

    key: str
    dtype: FeatureType = FeatureType('').float

    name: str = 'abs'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:
        from numpy import abs

        return abs(df[self.key])

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:
        return df.with_column(pl.col(self.key).abs().alias(alias))


@dataclass
class Mean(Transformation):

    key: str
    group_keys: list[str] | None = field(default=None)
    # sliding_window: float | None = field(default=None)
    name: str = 'mean'

    async def transform_pandas(self, df: pd.DataFrame) -> pd.Series:

        # df.set_index("event_timestamp").rolling(2).mean()

        if self.group_keys:
            if len(self.group_keys) == 1:
                group_key = self.group_keys[0]
                group_by_result = df.groupby(group_key)[self.key].mean()
                return df[group_key].map(group_by_result)
            else:
                raise ValueError('Group by with multiple keys is not suppported yet')
        else:
            return df[self.key].mean()

    async def transform_polars(self, df: pl.LazyFrame, alias: str) -> pl.LazyFrame:

        if self.group_keys:
            # if len(self.group_keys) == 1:
            #     group_key = self.group_keys[0]
            # return df.join(df.groupby(group_key).agg(pl.col(self.key).mean()
            # .alias("agg_mean")), on=group_key, how="left").select("agg_mean_right")
            raise ValueError('Group by is not supported for polars yet')
        else:
            return df.with_column(pl.col(self.key).mean().alias(alias))
