import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from pandas import DataFrame, Index, concat
from pandas.core.generic import NDFrame

DatasetType = TypeVar('DatasetType', bound=NDFrame)


@dataclass
class TrainTestSet(Generic[DatasetType]):

    data: DatasetType

    features: list[str]
    target: str

    train_index: Index
    test_index: Index

    @property
    def train(self) -> DatasetType:
        return self.data.iloc[self.train_index]

    @property
    def train_input(self) -> DatasetType:
        return self.train[self.features]

    @property
    def train_output(self) -> DatasetType:
        return self.train[self.target]

    @property
    def test(self) -> DatasetType:
        return self.data.iloc[self.test_index]

    @property
    def test_input(self) -> DatasetType:
        return self.test[self.features]

    @property
    def test_output(self) -> DatasetType:
        return self.test[self.target]


@dataclass
class SupervisedDataSet(Generic[DatasetType]):
    data: DatasetType

    features: list[str]
    target: str

    @property
    def input(self) -> DatasetType:
        return self.data[self.features]

    @property
    def output(self) -> DatasetType:
        return self.data[self.target]


@dataclass
class TrainTestValidateSet(Generic[DatasetType]):

    data: DatasetType

    features: list[str]
    target: str

    train_index: Index
    test_index: Index
    validate_index: Index

    @property
    def input(self) -> DatasetType:
        return self.data[self.features]

    @property
    def output(self) -> DatasetType:
        return self.data[self.target]

    @property
    def train(self) -> SupervisedDataSet[DatasetType]:
        return SupervisedDataSet(self.data.iloc[self.train_index], self.features, self.target)

    @property
    def train_input(self) -> DatasetType:
        return self.train.input

    @property
    def train_output(self) -> DatasetType:
        return self.train.output

    @property
    def test(self) -> SupervisedDataSet[DatasetType]:
        return SupervisedDataSet(self.data.iloc[self.test_index], self.features, self.target)

    @property
    def test_input(self) -> DatasetType:
        return self.test.input

    @property
    def test_output(self) -> DatasetType:
        return self.test.output

    @property
    def validate(self) -> SupervisedDataSet[DatasetType]:
        return SupervisedDataSet(self.data.iloc[self.validate_index], self.features, self.target)

    @property
    def validate_input(self) -> DatasetType:
        return self.validate.input

    @property
    def validate_output(self) -> DatasetType:
        return self.validate.output


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

            train = concat([train, split(sub_group, 0, self.train_size_percentage)], axis=0)
            test = concat(
                [
                    test,
                    split(
                        sub_group,
                        self.train_size_percentage,
                        self.train_size_percentage + self.test_size_percentage,
                    ),
                ],
                axis=0,
            )
            develop = concat(
                [develop, split(sub_group, self.train_size_percentage + self.test_size_percentage, 1)], axis=0
            )

        return SplitDataSet(
            train_input=train.drop(columns=[target_column]),
            train_output=train[target_column],
            develop_input=develop.drop(columns=[target_column]),
            develop_output=develop[target_column],
            test_input=test.drop(columns=[target_column]),
            test_output=test[target_column],
        )
