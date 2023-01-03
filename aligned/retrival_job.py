from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod, abstractproperty
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from aligned.exceptions import UnableToFindFileException
from aligned.request.retrival_request import RequestResult, RetrivalRequest
from aligned.schemas.feature import FeatureType
from aligned.split_strategy import SplitDataSet, SplitStrategy, TrainTestSet, TrainTestValidateSet
from aligned.validation.interface import Validator

if TYPE_CHECKING:
    from aligned.local.source import DataFileReference


logger = logging.getLogger(__name__)


def split(data: pd.DataFrame, target_column: str, start_ratio: float, end_ratio: float) -> pd.Index:
    index = pd.Index([], dtype=data.index.dtype)
    for value in data[target_column].unique():
        subset = data.loc[data[target_column] == value]
        group_size = subset.shape[0]
        start_index = round(group_size * start_ratio)
        end_index = round(group_size * end_ratio)

        if end_index >= group_size:
            index = index.append(subset.iloc[start_index:].index)
        else:
            index = index.append(subset.iloc[start_index:end_index].index)
    return index


class RetrivalJob(ABC):
    @abstractproperty
    def request_result(self) -> RequestResult:
        pass

    @abstractmethod
    async def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    async def to_polars(self) -> pl.LazyFrame:
        raise NotImplementedError()

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        if isinstance(location, str):
            from aligned.local.source import ParquetFileSource

            return FileCachedJob(ParquetFileSource(location), self)
        else:
            return FileCachedJob(location, self)

    def test_size(self, test_size: float, target_column: str) -> TrainTestSetJob:

        return TrainTestSetJob(job=self, test_size=test_size, target_column=target_column)

    def validate(self, validator: Validator) -> ValidationJob:

        return ValidationJob(self, validator)

    def derive_features(self, requests: list[RetrivalRequest]) -> RetrivalJob:
        return DerivedFeatureJob(job=self, requests=requests)

    def ensure_types(self, requests: list[RetrivalRequest]) -> RetrivalJob:
        return EnsureTypesJob(job=self, requests=requests)


@dataclass
class ValidationJob(RetrivalJob):

    job: RetrivalJob
    validator: Validator

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    async def to_pandas(self) -> pd.DataFrame:
        return await self.validator.validate_pandas(
            list(self.request_result.features), await self.job.to_pandas()
        )

    async def to_polars(self) -> pl.LazyFrame:
        raise NotImplementedError()

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        if isinstance(location, str):
            from aligned.local.source import ParquetFileSource

            return FileCachedJob(ParquetFileSource(location), self)
        else:
            return FileCachedJob(location, self)


@dataclass
class DerivedFeatureJob(RetrivalJob):

    job: RetrivalJob
    requests: list[RetrivalRequest]

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    async def compute_derived_features_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        for request in self.requests:
            for feature_round in request.derived_features_order():
                for feature in feature_round:
                    if feature.depending_on_views - {request.feature_view_name}:
                        continue
                    logger.info(f'Computing feature: {feature.name}')
                    df = await feature.transformation.transform_polars(df, feature.name)
        return df

    async def compute_derived_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        for request in self.requests:
            for feature_round in request.derived_features_order():
                for feature in feature_round:
                    if feature.depending_on_views - {request.feature_view_name}:
                        continue

                    logger.info(f'Computing feature: {feature.name}')
                    df[feature.name] = await feature.transformation.transform_pandas(
                        df[feature.depending_on_names]
                    )
                    if df[feature.name].dtype != feature.dtype.pandas_type:
                        if feature.dtype.is_numeric:
                            df[feature.name] = pd.to_numeric(df[feature.name], errors='coerce').astype(
                                feature.dtype.pandas_type
                            )
                        else:
                            df[feature.name] = df[feature.name].astype(feature.dtype.pandas_type)
        return df

    async def to_pandas(self) -> pd.DataFrame:
        return await self.compute_derived_features_pandas(await self.job.to_pandas())

    async def to_polars(self) -> pl.LazyFrame:
        return await self.compute_derived_features_polars(await self.job.to_polars())


@dataclass
class RawFileCachedJob(RetrivalJob):

    location: DataFileReference
    job: DerivedFeatureJob

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    async def to_pandas(self) -> pd.DataFrame:
        from aligned.local.job import FileFullJob
        from aligned.local.source import LiteralReference

        try:
            logger.info('Trying to read cache file')
            df = await self.location.read_pandas()
        except UnableToFindFileException:
            logger.info('Unable to load file, so fetching from source')
            df = await self.job.job.to_pandas()
            logger.info('Writing result to cache')
            await self.location.write_pandas(df)
        return (
            await FileFullJob(LiteralReference(df), request=self.job.requests[0])
            .derive_features(self.job.requests)
            .to_pandas()
        )

    async def to_polars(self) -> pl.LazyFrame:
        return await self.job.to_polars()

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        return self


@dataclass
class FileCachedJob(RetrivalJob):

    location: DataFileReference
    job: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    async def to_pandas(self) -> pd.DataFrame:
        try:
            logger.info('Trying to read cache file')
            df = await self.location.read_pandas()
        except UnableToFindFileException:
            logger.info('Unable to load file, so fetching from source')
            df = await self.job.to_pandas()
            logger.info('Writing result to cache')
            await self.location.write_pandas(df)
        return df

    async def to_polars(self) -> pl.LazyFrame:
        return await super().to_polars()

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        return self


@dataclass
class TrainTestSetJob:

    job: RetrivalJob
    test_size: float
    target_column: str

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    async def use_df(self) -> TrainTestSet[pd.DataFrame]:

        data = await self.job.to_pandas()

        features = list({feature.name for feature in self.request_result.features} - {self.target_column})

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

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

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

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    async def use_pandas(self) -> SplitDataSet[pd.DataFrame]:
        data = await self.job.to_pandas()
        return self.strategy.split_pandas(data, self.target_column)


class FullExtractJob(RetrivalJob):
    limit: int | None


class DateRangeJob(RetrivalJob):
    start_date: datetime
    end_date: datetime


class FactualRetrivalJob(RetrivalJob):
    facts: dict[str, list]


@dataclass
class EnsureTypesJob(RetrivalJob):

    job: RetrivalJob
    requests: list[RetrivalRequest]

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_pandas()
        for request in self.requests:
            for feature in request.all_required_features:

                mask = ~df[feature.name].isnull()

                with suppress(AttributeError):
                    df[feature.name] = df[feature.name].mask(
                        ~mask, other=df.loc[mask, feature.name].str.strip('"')
                    )

                if feature.dtype == FeatureType('').datetime:
                    df[feature.name] = pd.to_datetime(df[feature.name], infer_datetime_format=True, utc=True)
                elif feature.dtype == FeatureType('').datetime or feature.dtype == FeatureType('').string:
                    continue
                else:

                    if feature.dtype.is_numeric:
                        df[feature.name] = pd.to_numeric(df[feature.name], errors='coerce').astype(
                            feature.dtype.pandas_type
                        )
                    else:
                        df[feature.name] = df[feature.name].astype(feature.dtype.pandas_type)
        return df

    async def to_polars(self) -> pl.LazyFrame:
        return await self.job.to_polars()


@dataclass
class CombineFactualJob(RetrivalJob):
    """Computes features that depend on different retrical jobs

    The `job` therefore take in a list of jobs that output some data,
    and a `combined_requests` which defines the features depending on the data

    one example would be the following

    class SomeView(FeatureView):
        metadata = FeatureViewMetadata(
            name="some_view",
            batch_source=FileSource.csv_at("data.csv")
        )
        id = Entity(Int32())
        a = Int32()

    class OtherView(FeatureView):
        metadata = FeatureViewMetadata(
            name="other_view",
            batch_source=FileSource.parquet_at("other.parquet")
        )
        id = Entity(Int32())
        c = Int32()

    class Combined(CombinedFeatureView):
        metadata = CombinedMetadata(name="combined")

        some = SomeView()
        other = OtherView()

        added = some.a + other.c
    """

    jobs: list[RetrivalJob]
    combined_requests: list[RetrivalRequest]

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_result_list(
            [job.request_result for job in self.jobs]
        ) + RequestResult.from_request_list(self.combined_requests)

    async def combine_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for request in self.combined_requests:
            for feature in request.derived_features:
                logger.info(f'Computing feature: {feature.name}')
                df[feature.name] = await feature.transformation.transform_pandas(
                    df[feature.depending_on_names]
                )
        return df

    async def combine_polars_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        for request in self.combined_requests:
            for feature in request.derived_features:
                logger.info(f'Computing feature: {feature.name}')
                df = await feature.transformation.transform_polars(df, feature.name)
        return df

    async def to_pandas(self) -> pd.DataFrame:
        job_count = len(self.jobs)
        if job_count > 1:
            dfs = await asyncio.gather(*[job.to_pandas() for job in self.jobs])
            df = pd.concat(dfs, axis=1)
            combined = await self.combine_data(df)
            return combined.loc[:, ~df.columns.duplicated()].copy()
        elif job_count == 1:
            df = await self.jobs[0].to_pandas()
            return await self.combine_data(df)
        else:
            raise ValueError(
                'Have no jobs to fetch. This is probably an internal error.\n'
                'Please submit an issue, and describe how to reproduce it.\n'
                'Or maybe even submit a PR'
            )

    async def to_polars(self) -> pl.LazyFrame:
        dfs: list[pl.LazyFrame] = await asyncio.gather(*[job.to_polars() for job in self.jobs])

        df = dfs[0]

        for other_df in dfs[1:]:
            df = df.with_context(other_df).select(pl.all())

        # df = pl.concat(dfs_to_concat, how='horizontal')
        return await self.combine_polars_data(df)

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        return CombineFactualJob([job.cached_at(location) for job in self.jobs], self.combined_requests)


@dataclass
class FilterJob(RetrivalJob):

    include_features: set[str]
    job: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result.filter_features(self.include_features)

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_pandas()
        if self.include_features:
            total_list = list({ent.name for ent in self.request_result.entities}.union(self.include_features))
            return df[total_list]
        else:
            return df

    async def to_polars(self) -> pl.LazyFrame:
        df = await self.job.to_polars()
        if self.include_features:
            total_list = list({ent.name for ent in self.request_result.entities}.union(self.include_features))
            return df.select(total_list)
        else:
            return df

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:

        return FilterJob(self.include_features, self.job.cached_at(location))
