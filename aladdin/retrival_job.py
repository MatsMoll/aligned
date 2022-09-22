from __future__ import annotations

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Generic, TypeVar

import pandas as pd

from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import FeatureType
from aladdin.feature_view.feature_view import FeatureView
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.split_strategy import SplitDataSet, SplitStrategy, TrainTestSet, TrainTestValidateSet

try:
    import dask.dataframe as dd

    GenericDataFrame = TypeVar('GenericDataFrame', pd.DataFrame, dd.DataFrame)
except ModuleNotFoundError:
    GenericDataFrame = pd.DataFrame  # type: ignore

if TYPE_CHECKING:
    from aladdin.local.source import StorageFileReference


logger = logging.getLogger(__name__)


def split(data: pd.DataFrame, target_column: str, start_ratio: float, end_ratio: float) -> pd.Index:
    index = pd.Index([], dtype=data.index.dtype)
    for value in data[target_column].unique():
        subset = data.loc[data[target_column] == value]
        group_size = subset.shape[0]
        start_index = math.floor(group_size * start_ratio)
        end_index = math.floor(group_size * end_ratio)

        index = index.append(subset.iloc[start_index:end_index].index)
    return index


class RetrivalJob(ABC):
    @abstractmethod
    async def to_df(self) -> pd.DataFrame:
        pass

    @abstractmethod
    async def to_dask(self) -> dd.DataFrame:
        pass

    @abstractmethod
    def cache_at(self, path: str) -> RetrivalJob:
        pass

    async def as_dataset(self, file: StorageFileReference) -> pd.DataFrame:
        import io

        df = await self.to_df()
        # FIXME: Should be included into the feast lib, as a conveniance and reduce user error
        for col in df.columns:  # Can't input UUID's, so need to convert all ids to strings
            if '_id' in col:
                df[col] = df[col].astype(str)

        data_bytes = io.BytesIO()
        df.to_parquet(data_bytes)  # write to BytesIO buffer
        data_bytes.seek(0)
        await file.write(data_bytes.getvalue())
        return df

    def test_size(self, test_size: float, target_column: str) -> TrainTestSetJob:

        return TrainTestSetJob(job=self, test_size=test_size, target_column=target_column)

    def join(self, feature_view: FeatureView) -> RetrivalJob:

        feature_view.compile().entitiy_names


class DerivedFeatureJob(RetrivalJob):
    @abstractmethod
    async def compute_derived_features(self, df: GenericDataFrame) -> GenericDataFrame:
        pass

    @abstractmethod
    async def ensure_types(self, df: GenericDataFrame) -> GenericDataFrame:
        pass

    @abstractmethod
    async def _to_df(self) -> pd.DataFrame:
        pass

    @abstractmethod
    async def _to_dask(self) -> dd.DataFrame:
        pass

    async def to_df(self) -> pd.DataFrame:
        df = await self._to_df()
        df = await self.ensure_types(df)
        return await self.compute_derived_features(df)

    async def to_dask(self) -> dd.DataFrame:
        df = await self._to_dask()
        df = await self.ensure_types(df)
        return await self.compute_derived_features(df)

    def cache_at(self, path: str) -> RetrivalJob:
        return FileCachedJob(path, self)


class LazyJoinJob(RetrivalJob):

    original_job: RetrivalJob
    view_to_join: FeatureView

    def to_df(self) -> pd.DataFrame:
        # original_df = await self.original_job.to_df()
        self.view_to_join.compile().batch_data_source


@dataclass
class FileCachedJob(RetrivalJob):

    path: str
    job: DerivedFeatureJob

    async def to_df(self) -> pd.DataFrame:
        try:
            df = pd.read_parquet(self.path)
        except FileNotFoundError:
            df = await self.job._to_df()
            df.to_parquet(self.path)
        df = await self.job.ensure_types(df)
        return await self.job.compute_derived_features(df)

    async def to_dask(self) -> dd.DataFrame:
        raise NotImplementedError()

    def cache_at(self, path: str) -> RetrivalJob:
        return self


@dataclass
class TrainTestSetJob:

    job: RetrivalJob
    test_size: float
    target_column: str

    async def use_df(self) -> TrainTestSet[pd.DataFrame]:

        data = await self.job.to_df()

        features = list(set(data.columns) - {self.target_column})

        test_ratio_start = 1 - self.test_size
        return TrainTestSet(
            data=data,
            features=features,
            target=self.target_column,
            train_index=split(data, self.target_column, 0, test_ratio_start),
            test_index=split(data, self.target_column, test_ratio_start, 1),
        )

    def validation_size(self, validation_size: float) -> TrainTestValidateSetJob:
        return TrainTestValidateSetJob(self, validation_size)


@dataclass
class TrainTestValidateSetJob:

    job: TrainTestSetJob
    validation_size: float

    async def use_df(self) -> TrainTestValidateSet[pd.DataFrame]:
        test_set = await self.job.use_df()

        train_data = test_set.train

        validation_ratio_start = 1 - 1 / (1 - self.job.test_size) * self.validation_size

        return TrainTestValidateSet(
            data=test_set.data,
            features=test_set.features,
            target=test_set.target,
            train_index=split(train_data, test_set.target, 0, validation_ratio_start),
            test_index=test_set.test_index,
            validate_index=split(train_data, test_set.target, validation_ratio_start, 1),
        )


@dataclass
class SplitJob:

    job: RetrivalJob
    target_column: str
    strategy: SplitStrategy

    async def use_pandas(self) -> SplitDataSet[pd.DataFrame]:
        data = await self.job.to_df()
        return self.strategy.split_pandas(data, self.target_column)


Source = TypeVar('Source', bound=BatchDataSource)


class SingleSourceRetrivalJob(DerivedFeatureJob, Generic[Source]):
    source: Source
    request: RetrivalRequest

    async def compute_derived_features(self, df: GenericDataFrame) -> GenericDataFrame:
        for feature_round in self.request.derived_features_order():
            for feature in feature_round:
                df[feature.name] = await feature.transformation.transform(df[feature.depending_on_names])
        return df

    async def ensure_types(self, df: GenericDataFrame) -> GenericDataFrame:
        for feature in self.request.all_required_features:
            mask = ~df[feature.name].isnull()

            with suppress(AttributeError):
                df[feature.name] = df[feature.name].mask(
                    ~mask, other=df.loc[mask, feature.name].str.strip('"')
                )

            if feature.dtype == FeatureType('').datetime:
                if isinstance(df, pd.DataFrame):
                    df[feature.name] = pd.to_datetime(df[feature.name], infer_datetime_format=True, utc=True)
                else:
                    df[feature.name] = dd.to_datetime(df[feature.name], infer_datetime_format=True, utc=True)
            elif feature.dtype == FeatureType('').datetime or feature.dtype == FeatureType('').string:
                continue
            else:

                if feature.dtype.is_numeric:
                    df[feature.name] = pd.to_numeric(df[feature.name], errors='coerce')
                else:
                    df[feature.name] = df[feature.name].astype(feature.dtype.pandas_type)
        return df


class FullExtractJob(SingleSourceRetrivalJob):
    limit: int | None


class DateRangeJob(SingleSourceRetrivalJob):
    start_date: datetime
    end_date: datetime


class FactualRetrivalJob(DerivedFeatureJob):

    requests: list[RetrivalRequest]
    facts: dict[str, list]

    async def compute_derived_features(self, df: GenericDataFrame) -> GenericDataFrame:
        combined_views: list[DerivedFeature] = []
        for request in self.requests:
            for feature_round in request.derived_features_order():
                for feature in feature_round:
                    if feature.depending_on_views - {request.feature_view_name}:
                        combined_views.append(feature)
                        continue

                    df[feature.name] = await feature.transformation.transform(df[feature.depending_on_names])
                    if df.dtypes[feature.name] != feature.dtype.pandas_type:
                        if feature.dtype.is_numeric:
                            df[feature.name] = pd.to_numeric(df[feature.name], errors='coerce')
                        else:
                            df[feature.name] = df[feature.name].astype(feature.dtype.pandas_type)

        return df

    async def ensure_types(self, df: GenericDataFrame) -> GenericDataFrame:
        for request in self.requests:
            for feature in request.all_required_features:

                try:
                    if feature.dtype == FeatureType('').datetime:

                        if isinstance(df, pd.DataFrame):
                            df[feature.name] = pd.to_datetime(
                                df[feature.name], infer_datetime_format=True, utc=True, errors='coerce'
                            )
                        else:
                            df[feature.name] = dd.to_datetime(
                                df[feature.name], infer_datetime_format=True, utc=True
                            )
                    elif feature.dtype == FeatureType('').string:
                        continue
                    else:
                        if feature.dtype.is_numeric:
                            df[feature.name] = pd.to_numeric(df[feature.name], errors='coerce')
                        else:
                            df[feature.name] = df[feature.name].astype(feature.dtype.pandas_type)
                except ValueError as error:
                    logger.info(f'Unable to ensure type for {feature.name}, error: {error}')
                    continue
        return df


@dataclass
class CombineFactualJob(RetrivalJob):

    jobs: list[RetrivalJob]
    combined_requests: list[RetrivalRequest]

    async def combine_data(self, df: GenericDataFrame) -> GenericDataFrame:
        for request in self.combined_requests:
            for feature in request.derived_features:
                df[feature.name] = await feature.transformation.transform(df[feature.depending_on_names])
        return df

    async def to_df(self) -> pd.DataFrame:
        job_count = len(self.jobs)
        if job_count > 1:
            dfs = await asyncio.gather(*[job.to_df() for job in self.jobs])
            df = pd.concat(dfs, axis=1)
            return await self.combine_data(df)
        elif job_count == 1:
            df = await self.jobs[0].to_df()
            return await self.combine_data(df)
        else:
            raise ValueError(
                'Have no jobs to fetch. This is probably an internal error.\n'
                'Please submit an issue, and describe how to reproduce it.\n'
                'Or maybe even submit a PR'
            )

    async def to_dask(self) -> dd.DataFrame:
        dfs = await asyncio.gather(*[job.to_dask() for job in self.jobs])
        df = dd.concat(dfs, axis=1)
        return await self.combine_data(df)

    def cache_at(self, path: str) -> RetrivalJob:
        return CombineFactualJob([job.cache_at(path) for job in self.jobs], self.combined_requests)


@dataclass
class FilterJob(RetrivalJob):

    include_features: set[str]
    job: RetrivalJob

    async def to_df(self) -> pd.DataFrame:
        df = await self.job.to_df()
        if self.include_features:
            return df[list(self.include_features)]
        else:
            return df

    async def to_dask(self) -> dd.DataFrame:
        df = await self.job.to_dask()
        if self.include_features:
            return df[list(self.include_features)]
        else:
            return df

    def cache_at(self, path: str) -> RetrivalJob:

        return FilterJob(self.include_features, self.job.cache_at(path))
