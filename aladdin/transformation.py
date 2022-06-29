from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from mashumaro.types import SerializableType
from pandas import DataFrame, Series

from aladdin.codable import Codable
from aladdin.feature import FeatureType
from aladdin.psql.data_source import PostgreSQLConfig


@dataclass
class TransformationTestDefinition:
    transformation: 'Transformation'
    input: dict[str, list]
    output: list

    @property
    def input_df(self) -> DataFrame:
        return DataFrame(self.input)

    @property
    def output_series(self) -> Series:
        return Series(self.output)


def gracefull_transformation(
    df: DataFrame, is_valid_mask: Series, transformation: Callable[[DataFrame], Series]
) -> Series:
    result = Series(np.repeat(np.nan, repeats=is_valid_mask.shape[0]))
    result.loc[is_valid_mask] = transformation(df.loc[is_valid_mask])
    return result


class Transformation(Codable, SerializableType):
    name: str
    dtype: FeatureType

    async def transform(self, df: DataFrame) -> Series:
        pass

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
    async def run_transformation_test(cls) -> None:
        import numpy as np
        from numpy.testing import assert_almost_equal

        with suppress(NotImplementedError):
            test = cls.test_definition()
            output = await test.transformation.transform(test.input_df)
            if test.transformation.dtype == FeatureType('').bool:
                is_correct = np.all(output == test.output_series) | output.equals(test.output_series)
                assert is_correct, f'Output for {test.transformation.__class__.__name__} is not correct.,'
                '\nGot: {output},\nexpected: {test.output_series}'
            else:
                expected = test.output_series.to_numpy()
                output_np = output.to_numpy().astype('float')
                is_null = np.isnan(expected) & np.isnan(output_np)
                assert_almost_equal(expected[~is_null], output_np[~is_null])


class SupportedTransformations:

    types: dict[str, type[Transformation]]

    _shared: Optional['SupportedTransformations'] = None

    def __init__(self) -> None:
        self.types = {}
        from aladdin.feature_types import CustomTransformationV2

        for tran_type in [
            Equals,
            NotEquals,
            CustomTransformationV2,
            TimeSeriesTransformation,
            StandardScalingTransformation,
            Ratio,
            Contains,
            GreaterThen,
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
class Equals(Transformation):

    key: str
    value: str

    name: str = 'equals'
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    async def transform(self, df: DataFrame) -> Series:
        return df[self.key] == self.value

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Equals('x', 'Test'),
            input={'x': ['Hello', 'Test', 'nah', 'test', 'Test']},
            output=[False, True, False, False, True],
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

    async def transform(self, df: DataFrame) -> Series:
        return df[self.key] != self.value

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            NotEquals('x', 'Test'),
            input={'x': ['Hello', 'Test', 'nah', 'test', 'Test']},
            output=[True, False, True, True, False],
        )


@dataclass
class GreaterThen(Transformation):

    key: str
    value: float

    name: str = 'gt'
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str, value: float) -> None:
        self.key = key
        self.value = value

    async def transform(self, df: DataFrame) -> Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] > self.value,
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            GreaterThen(key='x', value=2), input={'x': [1, 2, 3, nan]}, output=[False, False, True, nan]
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

    async def transform(self, df: DataFrame) -> Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] >= self.value,
        )

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

    async def transform(self, df: DataFrame) -> Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | df[self.key].isnull()),
            transformation=lambda dfv: dfv[self.key] < self.value,
        )

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

    async def transform(self, df: DataFrame) -> Series:
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

    async def transform(self, df: DataFrame) -> Series:
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

    async def transform(self, df: DataFrame) -> Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.front].isna() | df[self.behind].isna()),
            transformation=lambda dfv: dfv[self.front] + dfv[self.behind],
        )

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

    async def transform(self, df: DataFrame) -> Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.front].isna() | df[self.behind].isna()),
            transformation=lambda dfv: (dfv[self.front] - dfv[self.behind]) / np.timedelta64(1, self.unit),
        )

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

    async def transform(self, df: DataFrame) -> Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | (df[self.key] <= 0)),
            transformation=lambda dfv: np.log(dfv[self.key]),
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

    async def transform(self, df: DataFrame) -> Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna() | (df[self.key] <= -1)),
            transformation=lambda dfv: np.log1p(dfv[self.key]),
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

    async def transform(self, df: DataFrame) -> Series:
        from pandas import to_numeric

        return to_numeric(df[self.key], errors='coerce')

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            ToNumerical('x'),
            input={'x': [1, '0', '10.5', None, nan, '-20']},
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

    async def transform(self, df: DataFrame) -> Series:
        from pandas import to_datetime

        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna()),
            transformation=lambda dfv: getattr(to_datetime(dfv[self.key]).dt, self.component),
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        from numpy import nan

        return TransformationTestDefinition(
            DateComponent(key='x', component='hour'),
            input={'x': ['2022-04-02T20:20:50', None, '2022-02-20T23:20:50', '1993-04-02T01:20:50']},
            output=[20, nan, 23, 1],
        )


@dataclass
class Contains(Transformation):

    key: str
    value: str

    name: str = 'contains'
    dtype: FeatureType = FeatureType('').bool

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    async def transform(self, df: DataFrame) -> Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(df[self.key].isna()),
            transformation=lambda dfv: dfv[self.key].str.contains(self.value),
        )

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            Contains('x', 'es'),
            input={'x': ['Hello', 'Test', 'nah', 'test', 'espania', None]},
            output=[False, True, False, True, True, None],
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

    async def transform(self, df: DataFrame) -> Series:
        return gracefull_transformation(
            df,
            is_valid_mask=~(
                df[self.numerator].isna() | df[self.denumerator].isna() | df[self.denumerator] == 0
            ),
            transformation=lambda dfv: dfv[self.numerator].astype(float)
            / dfv[self.denumerator].astype(float),
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
class TimeSeriesTransformation(Transformation):

    method: str
    field_name: str
    table_name: str
    config: PostgreSQLConfig
    event_timestamp_column: str

    dtype: FeatureType = FeatureType('').int64
    name: str = 'ts_transform'

    async def transform(self, df: DataFrame) -> Series:
        import numpy as np

        org_facts = df[[self.event_timestamp_column, self.field_name]]
        ret_data = Series(np.repeat(np.nan, org_facts.shape[0]))

        mask = org_facts.notna().all(axis=1)

        fact_df = org_facts.loc[mask]

        if fact_df.empty:
            return ret_data

        fact_df['row_id'] = list(range(1, fact_df.shape[0] + 1))

        values = []
        columns = ','.join(list(fact_df.columns))
        for _, row in fact_df.iterrows():
            row_values = []
            for column, value in row.items():
                if column == self.event_timestamp_column:
                    row_values.append(f"'{value}'::timestamp with time zone")
                else:
                    row_values.append(f"'{value}'")
            values.append(','.join(row_values))

        sql_values = '(' + '),\n    ('.join(values) + ')'
        sql = f"""
WITH entities (
    {columns}
) AS (
VALUES
    {sql_values}
)

SELECT {self.method}(t.{self.field_name}) AS {self.field_name}_value, et.row_id
FROM entities et
LEFT JOIN {self.table_name} t ON
    t.{self.field_name} = et.{self.field_name} AND
    t.{self.event_timestamp_column} < et.{self.event_timestamp_column}
GROUP BY et.row_id;
"""
        data = await self.config.data_enricher(sql).load()

        ret_data[mask] = data[f'{self.field_name}_value']
        return ret_data


@dataclass
class StandardScalingTransformation(Transformation):

    mean: float
    std: float
    key: str

    name = 'standard_scaling'
    dtype = FeatureType('').float

    async def transform(self, df: DataFrame) -> Series:
        return (df[self.key] - self.mean) / self.std

    @staticmethod
    def test_definition() -> TransformationTestDefinition:
        return TransformationTestDefinition(
            StandardScalingTransformation(mean=1, std=0.5, key='x'),
            input={'x': [1, 1.5, 0.5, 1, 2, 3]},
            output=[0, 1, -1, 0, 2, 4],
        )
