from dataclasses import dataclass
from typing import Optional

from mashumaro.types import SerializableType
from pandas import DataFrame, Series

from aladdin.codable import Codable
from aladdin.feature import FeatureType
from aladdin.psql.data_source import PostgreSQLConfig


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
        return data_class.from_dict(value)


class SupportedTransformations:

    types: dict[str, type[Transformation]]

    _shared: Optional['SupportedTransformations'] = None

    def __init__(self) -> None:
        self.types = {}
        from aladdin.feature_types import CustomTransformationV2

        for tran_type in [Equals, CustomTransformationV2, TimeSeriesTransformation]:
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
        fact_df = df[[self.event_timestamp_column, self.field_name]]
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
        return data[f'{self.field_name}_value']
