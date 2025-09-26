from __future__ import annotations
from typing import Generic, TypeVar, overload

import polars as pl
from aligned.lazy_imports import pandas as pd
from aligned.request.retrieval_request import RetrievalRequest
from aligned.schemas.feature import FeatureReference


DatasetType = TypeVar("DatasetType")


def column_for(
    data: pl.DataFrame | pl.LazyFrame | pd.DataFrame, column: str
) -> pl.Series | pd.Series:
    if isinstance(data, pl.DataFrame):
        return data[column]
    elif isinstance(data, pl.LazyFrame):
        return data.select(column).collect()[column]
    elif isinstance(data, pd.DataFrame):
        col: pd.Series = data[column]  # type: ignore
        return col

    raise NotImplementedError(f"Unsupported type {type(data)}")


class SupervisedDataSet(Generic[DatasetType]):
    data: DatasetType

    entity_columns: set[str]
    feature_columns: set[str]
    target_columns: set[str]
    event_timestamp_column: str | None
    requests: list[RetrievalRequest]

    @property
    def feature_references(self) -> list[FeatureReference]:
        refs = set()
        for request in self.requests:
            refs.update(
                feat.as_reference(request.location)
                for feat in request.all_features
                if feat.name not in self.target_columns
            )
        return sorted(refs, key=lambda ref: ref.name)

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
        requests: list[RetrievalRequest],
    ):
        self.data = data
        self.entity_columns = entity_columns
        self.feature_columns = features
        self.target_columns = target
        self.event_timestamp_column = event_timestamp_column
        self.requests = requests

    @property
    def entities(self) -> DatasetType:
        if isinstance(self.data, (pl.LazyFrame, pl.DataFrame)):
            return self.data.select(list(self.entity_columns))  # type: ignore
        return self.data[list(self.entity_columns)]  # type: ignore

    @property
    def input(self) -> DatasetType:
        if isinstance(self.data, (pl.LazyFrame, pl.DataFrame)):
            return self.data.select(self.sorted_features)  # type: ignore
        return self.data[self.sorted_features]  # type: ignore

    @overload
    def label(self: "SupervisedDataSet[pl.DataFrame]") -> pl.Series: ...

    @overload
    def label(self: "SupervisedDataSet[pl.LazyFrame]") -> pl.Series: ...

    @overload
    def label(self: "SupervisedDataSet[pd.DataFrame]") -> pd.Series: ...

    def label(self) -> pl.Series | pd.Series:
        assert len(self.target_columns) == 1
        return column_for(self.data, next(iter(self.target_columns)))  # type: ignore

    @property
    def labels(self) -> DatasetType:
        if isinstance(self.data, (pl.LazyFrame, pl.DataFrame)):
            return self.data.select(list(self.target_columns))  # type: ignore
        return self.data[list(self.target_columns)]  # type: ignore

    @overload
    def target(self: "SupervisedDataSet[pl.DataFrame]") -> pl.Series: ...

    @overload
    def target(self: "SupervisedDataSet[pl.LazyFrame]") -> pl.Series: ...

    @overload
    def target(self: "SupervisedDataSet[pd.DataFrame]") -> pd.Series: ...

    def target(self) -> pl.Series | pd.Series:
        assert len(self.target_columns) == 1
        return column_for(self.data, next(iter(self.target_columns)))  # type: ignore
