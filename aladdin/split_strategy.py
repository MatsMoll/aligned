import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from pandas import DataFrame

DatasetType = TypeVar('DatasetType')


@dataclass
class SplitDataSet(Generic[DatasetType]):

    train_input: DatasetType
    train_output: DatasetType

    develop_input: DatasetType
    develop_output: DatasetType

    test_input: DatasetType
    test_output: DatasetType


class SplitStrategy:
    def split_pandas(self, data: DataFrame, target_column: str) -> SplitDataSet[DataFrame]:
        pass


class StrategicSplitStrategy(SplitStrategy):

    train_size_percentage: float
    test_size_percentage: float

    def __init__(self, train_size_percentage: float, test_size_percentage: float):
        assert train_size_percentage + test_size_percentage <= 1
        self.train_size_percentage = train_size_percentage
        self.test_size_percentage = test_size_percentage

    def split_pandas(self, data: DataFrame, target_column: str) -> SplitDataSet[DataFrame]:
        train = DataFrame(columns=data.columns)
        test = DataFrame(columns=data.columns)
        develop = DataFrame(columns=data.columns)

        target_data = data[target_column]

        def split(data: DataFrame, start_ratio: float, end_ratio: float) -> DataFrame:
            group_size = data.shape[0]
            start_index = math.floor(group_size * start_ratio)
            end_index = math.floor(group_size * end_ratio)
            return data.iloc[start_index:end_index]

        for target in target_data.unique():
            sub_group = data.loc[target_data == target]

            train = train.append(split(sub_group, 0, self.train_size_percentage))
            test = test.append(
                split(
                    sub_group,
                    self.train_size_percentage,
                    self.train_size_percentage + self.test_size_percentage,
                )
            )
            develop = develop.append(
                split(sub_group, self.train_size_percentage + self.test_size_percentage, 1)
            )

        return SplitDataSet(
            train_input=train.drop(columns=[target_column]),
            train_output=train[target_column],
            develop_input=develop.drop(columns=[target_column]),
            develop_output=develop[target_column],
            test_input=test.drop(columns=[target_column]),
            test_output=test[target_column],
        )
