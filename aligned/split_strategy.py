from __future__ import annotations
from typing import Generic, TypeVar

import polars as pl
from aligned.lazy_imports import pandas as pd

DatasetType = TypeVar('DatasetType')


class SupervisedDataSet(Generic[DatasetType]):
    data: DatasetType

    entity_columns: set[str]
    feature_columns: set[str]
    target_columns: set[str]
    event_timestamp_column: str | None

    @property
    def sorted_features(self) -> list[str]:
        return sorted(self.feature_columns)

    def __init__(
        self,
        data: DatasetType,
        entity_columns: set[str],
        features: set[str],
        target: set[str],
        event_timestamp_column: str | None,
    ):
        self.data = data
        self.entity_columns = entity_columns
        self.feature_columns = features
        self.target_columns = target
        self.event_timestamp_column = event_timestamp_column

    @property
    def entities(self) -> DatasetType:
        if isinstance(self.data, (pl.LazyFrame, pl.DataFrame)):
            return self.data.select(list(self.entity_columns))
        return self.data[list(self.entity_columns)]  # type: ignore

    @property
    def input(self) -> DatasetType:
        if isinstance(self.data, (pl.LazyFrame, pl.DataFrame)):
            return self.data.select(self.sorted_features)
        return self.data[self.sorted_features]  # type: ignore

    @property
    def labels(self) -> DatasetType:
        if isinstance(self.data, (pl.LazyFrame, pl.DataFrame)):
            return self.data.select(list(self.target_columns))
        return self.data[list(self.target_columns)]  # type: ignore


class TrainTestValidateSet(Generic[DatasetType]):

    data: DatasetType

    entity_columns: set[str]
    feature_columns: set[str]
    target_columns: set[str]

    train_index: 'pd.Index'
    test_index: 'pd.Index'
    validate_index: 'pd.Index'
    event_timestamp_column: str | None

    def __init__(
        self,
        data: DatasetType,
        entity_columns: set[str],
        features: set[str],
        target: set[str],
        train_index: 'pd.Index',
        test_index: 'pd.Index',
        validate_index: 'pd.Index',
        event_timestamp_column: str | None,
    ):
        self.data = data
        self.entity_columns = entity_columns
        self.feature_columns = features
        self.target_columns = target
        self.train_index = train_index
        self.test_index = test_index
        self.validate_index = validate_index
        self.event_timestamp_column = event_timestamp_column

    @property
    def sorted_features(self) -> list[str]:
        return sorted(self.feature_columns)

    @property
    def input(self) -> DatasetType:
        if isinstance(self.data, pl.LazyFrame):
            return self.data.select(self.sorted_features)
        return self.data[self.sorted_features]  # type: ignore

    @property
    def labels(self) -> DatasetType:
        if isinstance(self.data, pl.LazyFrame):
            return self.data.select(sorted(self.target_columns))
        return self.data[sorted(self.target_columns)]  # type: ignore

    @property
    def train(self) -> SupervisedDataSet[DatasetType]:
        if isinstance(self.data, pl.DataFrame):
            data = self.data[self.train_index.to_list(), :]
        else:
            data = self.data.iloc[self.train_index]  # type: ignore

        return SupervisedDataSet(  # type: ignore
            data,
            self.entity_columns,
            self.feature_columns,
            self.target_columns,
            self.event_timestamp_column,
        )

    @property
    def train_input(self) -> DatasetType:
        return self.train.input

    @property
    def train_target(self) -> DatasetType:
        return self.train.labels

    @property
    def test(self) -> SupervisedDataSet[DatasetType]:

        if isinstance(self.data, pl.DataFrame):
            data = self.data[self.test_index.to_list(), :]
        else:
            data = self.data.iloc[self.test_index]  # type: ignore

        return SupervisedDataSet(  # type: ignore
            data,
            set(self.entity_columns),
            set(self.feature_columns),
            self.target_columns,
            self.event_timestamp_column,
        )

    @property
    def test_input(self) -> DatasetType:
        return self.test.input

    @property
    def test_target(self) -> DatasetType:
        return self.test.labels

    @property
    def validate(self) -> SupervisedDataSet[DatasetType]:
        if isinstance(self.data, pl.DataFrame):
            data = self.data[self.validate_index.to_list(), :]
        else:
            data = self.data.iloc[self.validate_index]  # type: ignore
        return SupervisedDataSet(  # type: ignore
            data,
            self.entity_columns,
            set(self.feature_columns),
            self.target_columns,
            self.event_timestamp_column,
        )

    @property
    def validate_input(self) -> DatasetType:
        return self.validate.input

    @property
    def validate_target(self) -> DatasetType:
        return self.validate.labels


class SplitDataSet(Generic[DatasetType]):

    train_input: DatasetType
    train_target: DatasetType

    develop_input: DatasetType
    develop_target: DatasetType

    test_input: DatasetType
    test_target: DatasetType

    def __init__(
        self,
        train_input: DatasetType,
        train_target: DatasetType,
        develop_input: DatasetType,
        develop_target: DatasetType,
        test_input: DatasetType,
        test_target: DatasetType,
    ):
        self.train_input = train_input
        self.train_target = train_target
        self.develop_input = develop_input
        self.develop_target = develop_target
        self.test_input = test_input
        self.test_target = test_target
