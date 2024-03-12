from __future__ import annotations
from aligned.schemas.date_formatter import DateFormatter

import asyncio
import logging
import timeit
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Union, TypeVar, Coroutine, Any

import pandas as pd
import polars as pl
from polars.type_aliases import TimeUnit
from prometheus_client import Histogram

from aligned.exceptions import UnableToFindFileException
from aligned.request.retrival_request import FeatureRequest, RequestResult, RetrivalRequest
from aligned.schemas.feature import Feature, FeatureType
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.vector_storage import VectorIndex
from aligned.split_strategy import (
    SplitDataSet,
    SplitStrategy,
    SupervisedDataSet,
    TrainTestSet,
    TrainTestValidateSet,
)
from aligned.validation.interface import Validator

if TYPE_CHECKING:
    from typing import AsyncIterator
    from aligned.schemas.folder import DatasetMetadata, DatasetStore

    from aligned.schemas.derivied_feature import AggregatedFeature, AggregateOver
    from aligned.schemas.model import EventTrigger
    from aligned.sources.local import DataFileReference, StorageFileReference
    from aligned.feature_source import WritableFeatureSource


logger = logging.getLogger(__name__)


def split(
    data: pd.DataFrame, start_ratio: float, end_ratio: float, event_timestamp_column: str | None = None
) -> pd.Index:
    index = pd.Index([], dtype=data.index.dtype)
    if event_timestamp_column:
        column = data[event_timestamp_column]
        if column.dtype != 'datetime64[ns]':
            column = pd.to_datetime(data[event_timestamp_column])
        data = data.iloc[column.sort_values().index]

    group_size = data.shape[0]
    start_index = round(group_size * start_ratio)
    end_index = round(group_size * end_ratio)

    if end_index >= group_size:
        index = index.append(data.iloc[start_index:].index)
    else:
        index = index.append(data.iloc[start_index:end_index].index)
    return index


def subset_polars(
    data: pl.DataFrame, start_ratio: float, end_ratio: float, event_timestamp_column: str | None = None
) -> pl.DataFrame:

    if event_timestamp_column:
        data = data.sort(event_timestamp_column)

    group_size = data.height
    start_index = round(group_size * start_ratio)
    end_index = round(group_size * end_ratio)

    if end_index >= group_size:
        return data[start_index:]
    else:
        return data[start_index:end_index]


def split_polars(
    data: pl.DataFrame, start_ratio: float, end_ratio: float, event_timestamp_column: str | None = None
) -> pd.Series:

    row_name = 'row_nr'
    data = data.with_row_count(row_name)

    if event_timestamp_column:
        data = data.sort(event_timestamp_column)
        # values = data.select(
        #     [
        #         pl.col(event_timestamp_column).quantile(start_ratio).alias('start_value'),
        #         pl.col(event_timestamp_column).quantile(end_ratio).alias('end_value'),
        #     ]
        # )
        # return data.filter(
        #     pl.col(event_timestamp_column).is_between(values[0, 'start_value'], values[0, 'end_value'])
        # ).collect()

    group_size = data.shape[0]
    start_index = round(group_size * start_ratio)
    end_index = round(group_size * end_ratio)

    if end_index >= group_size:
        return data[start_index:][row_name].to_pandas()
    else:
        return data[start_index:end_index][row_name].to_pandas()


@dataclass
class TrainTestJob:

    train_job: RetrivalJob
    test_job: RetrivalJob

    target_columns: set[str]

    @property
    def train(self) -> SupervisedJob:
        return SupervisedJob(self.train_job, self.target_columns)

    @property
    def test(self) -> SupervisedJob:
        return SupervisedJob(self.test_job, self.target_columns)

    def store_dataset(
        self,
        dataset_store: DatasetStore,
        metadata: DatasetMetadata,
        train_source: DataFileReference,
        test_source: DataFileReference,
        train_size: float | None = None,
        test_size: float | None = None,
    ) -> TrainTestJob:
        from aligned.schemas.folder import TrainDatasetMetadata
        from aligned.data_source.batch_data_source import BatchDataSource

        request_result = self.train_job.request_result

        if isinstance(self.train_job, SubsetJob):
            train_size = self.train_job.fraction

        if isinstance(self.test_job, SubsetJob):
            test_size = self.test_job.fraction

        if not isinstance(test_source, BatchDataSource):
            raise ValueError('test_source should be a BatchDataSource')

        if not isinstance(train_source, BatchDataSource):
            raise ValueError('train_source should be a BatchDataSource')

        test_metadata = TrainDatasetMetadata(
            id=metadata.id,
            name=metadata.name,
            request_result=request_result,
            description=metadata.description,
            train_size_fraction=train_size,
            test_size_fraction=test_size,
            train_dataset=train_source,
            test_dataset=test_source,
            target=list(self.target_columns),
        )

        async def update_metadata() -> None:
            await dataset_store.store_train_test(test_metadata)

        return TrainTestJob(
            train_job=self.train_job.on_load(update_metadata).cached_at(train_source),
            test_job=self.test_job.on_load(update_metadata).cached_at(test_source),
            target_columns=self.target_columns,
        )


@dataclass
class TrainTestValidateJob:

    train_job: RetrivalJob
    test_job: RetrivalJob
    validate_job: RetrivalJob

    target_columns: set[str]

    @property
    def train(self) -> SupervisedJob:
        return SupervisedJob(self.train_job, self.target_columns)

    @property
    def test(self) -> SupervisedJob:
        return SupervisedJob(self.test_job, self.target_columns)

    @property
    def validate(self) -> SupervisedJob:
        return SupervisedJob(self.validate_job, self.target_columns)

    def store_dataset(
        self,
        dataset_store: DatasetStore | StorageFileReference,
        metadata: DatasetMetadata,
        train_source: DataFileReference,
        test_source: DataFileReference,
        validate_source: DataFileReference,
        train_size: float | None = None,
        test_size: float | None = None,
        validation_size: float | None = None,
    ) -> TrainTestValidateJob:
        from aligned.schemas.folder import TrainDatasetMetadata, JsonDatasetStore
        from aligned.data_source.batch_data_source import BatchDataSource
        from aligned.sources.local import StorageFileReference

        if isinstance(dataset_store, StorageFileReference):
            data_store = JsonDatasetStore(dataset_store)
        else:
            data_store = dataset_store

        request_result = self.train_job.request_result

        if isinstance(self.train_job, SubsetJob):
            train_size = self.train_job.fraction

        if isinstance(self.test_job, SubsetJob):
            test_size = self.test_job.fraction

        if isinstance(self.validate_job, SubsetJob):
            validation_size = self.validate_job.fraction

        if not isinstance(test_source, BatchDataSource):
            raise ValueError('test_source should be a BatchDataSource')

        if not isinstance(train_source, BatchDataSource):
            raise ValueError('train_source should be a BatchDataSource')

        if not isinstance(validate_source, BatchDataSource):
            raise ValueError('validation_source should be a BatchDataSource')

        test_metadata = TrainDatasetMetadata(
            id=metadata.id,
            name=metadata.name,
            request_result=request_result,
            description=metadata.description,
            train_size_fraction=train_size,
            test_size_fraction=test_size,
            validate_size_fraction=validation_size,
            train_dataset=train_source,
            test_dataset=test_source,
            validation_dataset=validate_source,
            target=list(self.target_columns),
        )

        async def update_metadata() -> None:
            await data_store.store_train_test_validate(test_metadata)

        return TrainTestValidateJob(
            train_job=self.train_job.cached_at(train_source).on_load(update_metadata).cached_at(train_source),
            test_job=self.test_job.cached_at(test_source).on_load(update_metadata).cached_at(test_source),
            validate_job=self.validate_job.cached_at(validate_source)
            .on_load(update_metadata)
            .cached_at(validate_source),
            target_columns=self.target_columns,
        )


@dataclass
class SupervisedJob:

    job: RetrivalJob
    target_columns: set[str]
    should_filter_out_null_targets: bool = True

    async def to_pandas(self) -> SupervisedDataSet[pd.DataFrame]:
        data = await self.job.to_pandas()

        if self.should_filter_out_null_targets:
            data = data.dropna(subset=list(self.target_columns))

        features = {
            feature.name
            for feature in self.job.request_result.features
            if feature.name not in self.target_columns
        }
        entities = {feature.name for feature in self.job.request_result.entities}
        return SupervisedDataSet(
            data, entities, features, self.target_columns, self.job.request_result.event_timestamp
        )

    async def to_polars(self) -> SupervisedDataSet[pl.DataFrame]:
        dataset = await self.to_lazy_polars()

        return SupervisedDataSet(
            data=dataset.data.collect(),
            entity_columns=dataset.entity_columns,
            features=dataset.feature_columns,
            target=dataset.target_columns,
            event_timestamp_column=dataset.event_timestamp_column,
        )

    async def to_lazy_polars(self) -> SupervisedDataSet[pl.LazyFrame]:
        data = await self.job.to_lazy_polars()
        if self.should_filter_out_null_targets:
            data = data.drop_nulls([column for column in self.target_columns])

        features = [
            feature.name
            for feature in self.job.request_result.features
            if feature.name not in self.target_columns
        ]
        entities = [feature.name for feature in self.job.request_result.entities]
        return SupervisedDataSet(
            data, set(entities), set(features), self.target_columns, self.job.request_result.event_timestamp
        )

    def should_filter_null_targets(self, should_filter: bool) -> SupervisedJob:
        self.should_filter_out_null_targets = should_filter
        return self

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    def train_set(self, train_size: float) -> SupervisedTrainJob:
        return SupervisedTrainJob(self, train_size)

    def train_test(self, train_size: float) -> TrainTestJob:

        cached_job = InMemoryCacheJob(self.job)

        event_timestamp = self.job.request_result.event_timestamp

        return TrainTestJob(
            train_job=SubsetJob(cached_job, 0, train_size, event_timestamp),
            test_job=SubsetJob(cached_job, train_size, 1, event_timestamp),
            target_columns=self.target_columns,
        )

    def train_test_validate(self, train_size: float, validate_size: float) -> TrainTestValidateJob:

        cached_job = InMemoryCacheJob(self.job)

        event_timestamp = self.job.request_result.event_timestamp

        test_ratio_start = train_size
        validate_ratio_start = test_ratio_start + validate_size

        return TrainTestValidateJob(
            train_job=SubsetJob(cached_job, 0, train_size, event_timestamp),
            test_job=SubsetJob(cached_job, train_size, validate_ratio_start, event_timestamp),
            validate_job=SubsetJob(cached_job, validate_ratio_start, 1, event_timestamp),
            target_columns=self.target_columns,
        )

    def with_subfeatures(self) -> SupervisedJob:
        return SupervisedJob(self.job.with_subfeatures(), self.target_columns)

    def cache_raw_data(self, location: DataFileReference | str) -> SupervisedJob:
        return SupervisedJob(
            self.job.cache_raw_data(location),
            self.target_columns,
        )

    def cached_at(self, location: DataFileReference | str) -> SupervisedJob:
        return SupervisedJob(
            self.job.cached_at(location),
            self.target_columns,
        )

    def drop_invalid(self, validator: Validator) -> SupervisedJob:
        return SupervisedJob(
            self.job.drop_invalid(validator),
            self.target_columns,
        )

    def log_each_job(self) -> SupervisedJob:
        return SupervisedJob(
            self.job.log_each_job(),
            self.target_columns,
        )

    def describe(self) -> str:
        return f'{self.job.describe()} with target columns {self.target_columns}'


@dataclass
class SupervisedTrainJob:

    job: SupervisedJob
    train_size: float

    async def to_pandas(self) -> TrainTestSet[pd.DataFrame]:
        core_data = await self.job.to_lazy_polars()
        data = core_data.data.collect()
        data = data.to_pandas()

        test_ratio_start = self.train_size
        return TrainTestSet(
            data=data,
            entity_columns=core_data.entity_columns,
            features=core_data.feature_columns,
            target_columns=core_data.target_columns,
            train_index=split(data, 0, test_ratio_start, core_data.event_timestamp_column),
            test_index=split(data, test_ratio_start, 1, core_data.event_timestamp_column),
            event_timestamp_column=core_data.event_timestamp_column,
        )

    async def to_polars(self) -> TrainTestSet[pl.DataFrame]:
        # Use the pandas method, as the split is not created for polars yet
        # A but unsure if I should use the same index concept for polars
        core_data = await self.job.to_lazy_polars()

        data = core_data.data.collect()

        return TrainTestSet(
            data=data,
            entity_columns=core_data.entity_columns,
            features=core_data.feature_columns,
            target_columns=core_data.target_columns,
            train_index=split_polars(data, 0, self.train_size, core_data.event_timestamp_column),
            test_index=split_polars(data, self.train_size, 1, core_data.event_timestamp_column),
            event_timestamp_column=core_data.event_timestamp_column,
        )

    def validation_set(self, validation_size: float) -> SupervisedValidationJob:
        return SupervisedValidationJob(self, validation_size)


@dataclass
class SupervisedValidationJob:

    job: SupervisedTrainJob
    validation_size: float

    async def to_pandas(self) -> TrainTestValidateSet[pd.DataFrame]:
        data = await self.job.to_pandas()

        test_start = self.job.train_size
        validate_start = test_start + self.validation_size

        return TrainTestValidateSet(
            data=data.data,
            entity_columns=set(data.entity_columns),
            features=data.features,
            target=data.target_columns,
            train_index=split(data.data, 0, test_start, data.event_timestamp_column),
            test_index=split(data.data, test_start, validate_start, data.event_timestamp_column),
            validate_index=split(data.data, validate_start, 1, data.event_timestamp_column),
            event_timestamp_column=data.event_timestamp_column,
        )

    async def to_polars(self) -> TrainTestValidateSet[pl.DataFrame]:
        data = await self.to_pandas()

        return TrainTestValidateSet(
            data=pl.from_pandas(data.data),
            entity_columns=data.entity_columns,
            features=data.feature_columns,
            target=data.labels,
            train_index=data.train_index,
            test_index=data.test_index,
            validate_index=data.validate_index,
            event_timestamp_column=data.event_timestamp_column,
        )


ConvertableToRetrivalJob = Union[dict[str, list], pd.DataFrame, pl.DataFrame, pl.LazyFrame]


class RetrivalJob(ABC):
    @property
    def request_result(self) -> RequestResult:
        if isinstance(self, ModificationJob):
            return self.job.request_result
        raise NotImplementedError(f'For {type(self)}')

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        if isinstance(self, ModificationJob):
            return self.job.retrival_requests
        raise NotImplementedError(f'For {type(self)}')

    @abstractmethod
    async def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError(f'For {type(self)}')

    @abstractmethod
    async def to_lazy_polars(self) -> pl.LazyFrame:
        raise NotImplementedError(f'For {type(self)}')

    async def to_polars(self) -> pl.DataFrame:
        return await (await self.to_lazy_polars()).collect_async()

    def describe(self) -> str:
        if isinstance(self, ModificationJob):
            return f'{self.job.describe()} -> {self.__class__.__name__}'
        raise NotImplementedError(f'Describe not implemented for {self.__class__.__name__}')

    def remove_derived_features(self) -> RetrivalJob:
        if isinstance(self, ModificationJob):
            return self.copy_with(self.job.remove_derived_features())
        return self

    def log_each_job(self) -> RetrivalJob:
        if isinstance(self, ModificationJob):
            return LogJob(self.copy_with(self.job.log_each_job()))
        return LogJob(self)

    def join_asof(
        self,
        job: RetrivalJob,
        left_event_timestamp: str | None = None,
        right_event_timestamp: str | None = None,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        timestamp_unit: TimeUnit = 'us',
    ) -> RetrivalJob:

        if isinstance(left_on, str):
            left_on = [left_on]

        if isinstance(right_on, str):
            right_on = [right_on]

        if not left_event_timestamp:
            left_event_timestamp = self.request_result.event_timestamp
        if not right_event_timestamp:
            right_event_timestamp = job.request_result.event_timestamp

        if not left_event_timestamp:
            raise ValueError('Missing an event_timestamp for left request.')
        if not right_event_timestamp:
            raise ValueError('Missing an event_timestamp for right request.')

        return JoinAsofJob(
            left_job=self,
            right_job=job,
            left_event_timestamp=left_event_timestamp,
            right_event_timestamp=right_event_timestamp,
            left_on=left_on,
            right_on=right_on,
            timestamp_unit=timestamp_unit,
        )

    def join(
        self, job: RetrivalJob, method: str, left_on: str | list[str], right_on: str | list[str]
    ) -> RetrivalJob:

        if isinstance(left_on, str):
            left_on = [left_on]

        if isinstance(right_on, str):
            right_on = [right_on]

        return JoinJobs(method=method, left_job=self, right_job=job, left_on=left_on, right_on=right_on)

    def on_load(self, on_load: Callable[[], Coroutine[Any, Any, None]]) -> RetrivalJob:
        return OnLoadJob(self, on_load)

    def filter(self, condition: str | Feature | DerivedFeature) -> RetrivalJob:
        return FilteredJob(self, condition)

    def chuncked(self, size: int) -> DataLoaderJob:
        return DataLoaderJob(self, size)

    def with_subfeatures(self) -> RetrivalJob:
        if isinstance(self, ModificationJob):
            return self.copy_with(self.job.with_subfeatures())
        return self

    def cache_raw_data(self, location: DataFileReference | str) -> RetrivalJob:
        if isinstance(self, ModificationJob):
            return self.copy_with(self.job.cache_raw_data(location))
        return self.cached_at(location)

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        if isinstance(location, str):
            from aligned.sources.local import ParquetFileSource

            return FileCachedJob(ParquetFileSource(location), self).derive_features()
        else:
            return FileCachedJob(location, self).derive_features()

    def test_size(self, test_size: float, target_column: str) -> SupervisedTrainJob:
        return SupervisedJob(self, {target_column}).train_set(train_size=1 - test_size)

    def train_set(self, train_size: float, target_column: str) -> SupervisedTrainJob:
        return SupervisedJob(self, {target_column}).train_set(train_size=train_size)

    def train_test(self, train_size: float, target_column: str) -> TrainTestJob:
        cached = InMemoryCacheJob(self)

        event_timestamp = self.request_result.event_timestamp

        return TrainTestJob(
            train_job=SubsetJob(cached, 0, train_size, event_timestamp),
            test_job=SubsetJob(cached, train_size, 1, event_timestamp),
            target_columns={target_column},
        )

    def train_test_validate(
        self, train_size: float, validate_size: float, target_column: str
    ) -> TrainTestValidateJob:

        cached = InMemoryCacheJob(self)

        event_timestamp = self.request_result.event_timestamp

        validate_ratio_start = train_size + validate_size

        return TrainTestValidateJob(
            train_job=SubsetJob(cached, 0, train_size, event_timestamp),
            test_job=SubsetJob(cached, train_size, validate_ratio_start, event_timestamp),
            validate_job=SubsetJob(cached, validate_ratio_start, 1, event_timestamp),
            target_columns={target_column},
        )

    def drop_invalid(self, validator: Validator) -> RetrivalJob:
        return DropInvalidJob(self, validator)

    def monitor_time_used(self, time_metric: Histogram, labels: list[str] | None = None) -> RetrivalJob:
        return TimeMetricLoggerJob(self, time_metric, labels)

    def derive_features(self, requests: list[RetrivalRequest] | None = None) -> RetrivalJob:
        requests = requests or self.retrival_requests

        for request in requests:
            if len(request.derived_features) > 0:
                return DerivedFeatureJob(job=self, requests=requests)
        return self

    def combined_features(self, requests: list[RetrivalRequest] | None = None) -> RetrivalJob:
        return CombineFactualJob([self], requests or self.retrival_requests)

    def ensure_types(
        self, requests: list[RetrivalRequest] | None = None, date_formatter: DateFormatter | None = None
    ) -> RetrivalJob:
        if not requests:
            requests = self.retrival_requests

        return EnsureTypesJob(
            job=self, requests=requests, date_formatter=date_formatter or DateFormatter.iso_8601()
        )

    def select_columns(self, include_features: set[str]) -> RetrivalJob:
        return SelectColumnsJob(include_features, self)

    def aggregate(self, request: RetrivalRequest) -> RetrivalJob:
        if not request.aggregated_features:
            return self
        return AggregateJob(self, request)

    def with_request(self, requests: list[RetrivalRequest]) -> RetrivalJob:
        return WithRequests(self, requests)

    def listen_to_events(self, events: set[EventTrigger]) -> RetrivalJob:
        return ListenForTriggers(self, events)

    def update_vector_index(self, indexes: list[VectorIndex]) -> RetrivalJob:
        return UpdateVectorIndexJob(self, indexes)

    def validate_entites(self) -> RetrivalJob:
        return ValidateEntitiesJob(self)

    def unique_on(self, unique_on: list[str], sort_key: str | None = None) -> RetrivalJob:
        return UniqueRowsJob(job=self, unique_on=unique_on, sort_key=sort_key)

    def unique_entities(self) -> RetrivalJob:
        request = self.request_result

        if not request.event_timestamp:
            logger.info(
                'Unable to find event_timestamp for `unique_entities`. '
                'This can lead to inconsistent features.'
            )

        return self.unique_on(unique_on=request.entity_columns, sort_key=request.event_timestamp)

    def fill_missing_columns(self) -> RetrivalJob:
        return FillMissingColumnsJob(self)

    def rename(self, mappings: dict[str, str]) -> RetrivalJob:
        if not mappings:
            return self
        return RenameJob(self, mappings)

    def drop_duplicate_entities(self) -> RetrivalJob:
        return DropDuplicateEntities(self)

    def ignore_event_timestamp(self) -> RetrivalJob:
        if isinstance(self, ModificationJob):
            return self.copy_with(self.job.ignore_event_timestamp())
        raise NotImplementedError('Not implemented ignore_event_timestamp')

    def polars_method(self, polars_method: Callable[[pl.LazyFrame], pl.LazyFrame]) -> RetrivalJob:
        return CustomPolarsJob(self, polars_method)

    @staticmethod
    def from_dict(data: dict[str, list], request: list[RetrivalRequest] | RetrivalRequest) -> RetrivalJob:
        if isinstance(request, RetrivalRequest):
            request = [request]
        return LiteralDictJob(data, request)

    @staticmethod
    def from_polars_df(df: pl.DataFrame, request: list[RetrivalRequest]) -> RetrivalJob:
        from aligned.local.job import LiteralRetrivalJob

        return LiteralRetrivalJob(df.lazy(), request)

    @staticmethod
    def from_convertable(
        data: ConvertableToRetrivalJob, request: list[RetrivalRequest] | RetrivalRequest | FeatureRequest
    ) -> RetrivalJob:
        from aligned.local.job import LiteralRetrivalJob

        if isinstance(request, RetrivalRequest):
            request = [request]
        elif isinstance(request, FeatureRequest):
            request = request.needed_requests

        if isinstance(data, dict):
            return LiteralDictJob(data, request)

        if isinstance(data, pl.DataFrame):
            return LiteralRetrivalJob(data.lazy(), request)
        elif isinstance(data, pl.LazyFrame):
            return LiteralRetrivalJob(data, request)
        elif isinstance(data, pd.DataFrame):
            return LiteralRetrivalJob(pl.from_pandas(data).lazy(), request)
        else:
            raise ValueError(f'Unable to convert {type(data)} to RetrivalJob')

    async def write_to_source(self, source: WritableFeatureSource | DataFileReference) -> None:
        """
        Writes the output of the retrival job to the passed source.

        ```python
        redis_cluster = RedisConfig.localhost()

        store = FeatureStore.from_dir(".")

        await (store.model("taxi")
            .all_predictions() # Reads predictions from a `prediction_source`
            .write_to_source(redis_cluster)
        )

        ```

        Args:
            source (WritableFeatureSource): A source that we can write to.
        """
        from aligned.sources.local import DataFileReference

        if isinstance(source, DataFileReference):
            await source.write_polars(await self.to_lazy_polars())
        else:
            await source.insert(self, self.retrival_requests)


JobType = TypeVar('JobType')


class ModificationJob:

    job: RetrivalJob

    def copy_with(self: JobType, job: RetrivalJob) -> JobType:
        self.job = job  # type: ignore
        return self


@dataclass
class CustomPolarsJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    polars_method: Callable[[pl.LazyFrame], pl.LazyFrame]  # type: ignore

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()
        return self.polars_method(df)

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_lazy_polars()
        return df.collect().to_pandas()


@dataclass
class SubsetJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    start_ratio: float
    end_ratio: float
    sort_column: str | None = None

    @property
    def fraction(self) -> float:
        return self.end_ratio - self.start_ratio

    async def to_lazy_polars(self) -> pl.LazyFrame:
        data = (await self.job.to_lazy_polars()).collect()
        return subset_polars(data, self.start_ratio, self.end_ratio, self.sort_column).lazy()

    async def to_pandas(self) -> pd.DataFrame:
        data = await self.job.to_pandas()
        selection = split(data, self.start_ratio, self.end_ratio, self.sort_column)
        return data.iloc[selection]


@dataclass
class OnLoadJob(RetrivalJob, ModificationJob):  # type: ignore

    job: RetrivalJob  # type
    on_load: Callable[[], Coroutine[Any, Any, None]]  # type: ignore

    async def to_pandas(self) -> pd.DataFrame:
        data = await self.job.to_pandas()
        await self.on_load()
        return data

    async def to_lazy_polars(self) -> pl.LazyFrame:
        data = (await self.job.to_lazy_polars()).collect()
        await self.on_load()
        return data.lazy()

    def describe(self) -> str:
        return f'OnLoadJob {self.on_load} -> {self.job.describe()}'


@dataclass
class EncodeDatesJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    formatter: DateFormatter
    columns: list[str]

    async def to_lazy_polars(self) -> pl.LazyFrame:
        data = await self.job.to_lazy_polars()
        return data.with_columns([self.formatter.encode_polars(column) for column in self.columns])

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()


@dataclass
class InMemoryCacheJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    cached_data: pl.DataFrame | None = None

    async def to_lazy_polars(self) -> pl.LazyFrame:
        if self.cached_data is not None:
            return self.cached_data.lazy()

        data = (await self.job.to_lazy_polars()).collect()
        self.cached_data = data
        return data.lazy()

    async def to_pandas(self) -> pd.DataFrame:
        if self.cached_data is not None:
            return self.cached_data.to_pandas()

        data = await self.job.to_pandas()
        self.cached_data = pl.from_pandas(data)
        return data


@dataclass
class AggregateJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    agg_request: RetrivalRequest

    @property
    def request_result(self) -> RequestResult:
        return self.agg_request.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return [self.agg_request]

    async def to_lazy_polars(self) -> pl.LazyFrame:
        from aligned.local.job import aggregate

        core_frame = await self.job.to_lazy_polars()

        existing_cols = set(core_frame.columns)
        agg_features = {agg.name for agg in self.agg_request.aggregated_features}
        missing_features = agg_features - existing_cols

        if not missing_features:
            logger.debug(f'Skipping aggregation of {agg_features}. Already existed.')
            return core_frame
        else:
            logger.debug(f'Aggregating {agg_features}, missing {missing_features}.')
            return await aggregate(self.agg_request, core_frame)

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    def describe(self) -> str:
        return f'Aggregating over {self.job.describe()}'


@dataclass
class JoinAsofJob(RetrivalJob):

    left_job: RetrivalJob
    right_job: RetrivalJob

    left_event_timestamp: str
    right_event_timestamp: str

    left_on: list[str] | None
    right_on: list[str] | None

    timestamp_unit: TimeUnit = field(default='us')

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_result_list([self.left_job.request_result, self.right_job.request_result])

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return RetrivalRequest.combine(self.left_job.retrival_requests + self.right_job.retrival_requests)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        left = await self.left_job.to_lazy_polars()
        right = await self.right_job.to_lazy_polars()

        return left.with_columns(
            pl.col(self.left_event_timestamp).dt.cast_time_unit(self.timestamp_unit),
        ).join_asof(
            right.with_columns(pl.col(self.right_event_timestamp).dt.cast_time_unit(self.timestamp_unit)),
            by_left=self.left_on,
            by_right=self.right_on,
            left_on=self.left_event_timestamp,
            right_on=self.right_event_timestamp,
        )

    def log_each_job(self) -> RetrivalJob:
        sub_log = JoinAsofJob(
            left_job=self.left_job.log_each_job(),
            right_job=self.right_job.log_each_job(),
            left_event_timestamp=self.left_event_timestamp,
            right_event_timestamp=self.right_event_timestamp,
            left_on=self.left_on,
            right_on=self.right_on,
        )
        return LogJob(sub_log)

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    def describe(self) -> str:
        return (
            f'({self.left_job.describe()}) -> '
            f'Joining on time {self.left_event_timestamp} with {self.left_on} and '
            f'{self.right_event_timestamp} and {self.right_on} ({self.right_job.describe()})'
        )


@dataclass
class JoinJobs(RetrivalJob):

    method: str
    left_job: RetrivalJob
    right_job: RetrivalJob

    left_on: list[str]
    right_on: list[str]

    @property
    def request_result(self) -> RequestResult:
        request = RequestResult.from_result_list(
            [self.left_job.request_result, self.right_job.request_result]
        )

        right_entities = self.right_job.request_result.entities

        for feature in right_entities:
            if feature.name in self.right_on:
                request.entities.remove(feature)

        return request

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return RetrivalRequest.combine(self.left_job.retrival_requests + self.right_job.retrival_requests)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        left = await self.left_job.to_lazy_polars()
        right = await self.right_job.to_lazy_polars()

        return_request = self.left_job.request_result

        # Need to ensure that the data types are the same. Otherwise will the join fail
        for left_col, right_col in zip(self.left_on, self.right_on):
            polars_type = [
                feature
                for feature in return_request.features.union(return_request.entities)
                if feature.name == left_col
            ]
            if not polars_type:
                raise ValueError(f'Unable to find {left_col} in left request {return_request}.')

            polars_type = polars_type[0].dtype.polars_type

            left_column_dtypes = dict(zip(left.columns, left.dtypes))
            right_column_dtypes = dict(zip(right.columns, right.dtypes))

            if not left_column_dtypes[left_col].is_(polars_type):
                left = left.with_columns(pl.col(left_col).cast(polars_type))

            if not right_column_dtypes[right_col].is_(polars_type):
                right = right.with_columns(pl.col(right_col).cast(polars_type))

        return left.join(right, left_on=self.left_on, right_on=self.right_on, how=self.method)

    def log_each_job(self) -> RetrivalJob:
        sub_log = JoinJobs(
            method=self.method,
            left_job=self.left_job.log_each_job(),
            right_job=self.right_job.log_each_job(),
            left_on=self.left_on,
            right_on=self.right_on,
        )
        return LogJob(sub_log)

    async def to_pandas(self) -> pd.DataFrame:
        left = await self.left_job.to_pandas()
        right = await self.right_job.to_pandas()
        return left.merge(right, how=self.method, left_on=self.left_on, right_on=self.right_on)

    def describe(self) -> str:
        return (
            f'({self.left_job.describe()}) -> '
            f'Joining with {self.method} {self.left_on} and '
            f'{self.right_on} ({self.right_job.describe()})'
        )


@dataclass
class FilteredJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    condition: DerivedFeature | Feature | str

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()

        if isinstance(self.condition, str):
            col = pl.col(self.condition)
        elif isinstance(self.condition, DerivedFeature):
            expr = await self.condition.transformation.transform_polars(df, self.condition.name)
            if isinstance(expr, pl.Expr):
                col = expr
            else:
                col = pl.col(self.condition.name)
        elif isinstance(self.condition, Feature):
            col = pl.col(self.condition.name)
        else:
            raise ValueError()

        return df.filter(col)

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_pandas()

        if isinstance(self.condition, str):
            mask = df[self.condition]
        elif isinstance(self.condition, DerivedFeature):
            mask = await self.condition.transformation.transform_pandas(df)
        elif isinstance(self.condition, Feature):
            mask = df[self.condition.name]
        else:
            raise ValueError()

        return df.loc[mask]

    def describe(self) -> str:
        return f'{self.job.describe()} -> Filter based on {self.condition}'


class JoinBuilder:

    joins: list[str]


@dataclass
class RenameJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    mappings: dict[str, str]

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_pandas()
        return df.rename(self.mappings)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()
        return df.rename(self.mappings)


@dataclass
class DropDuplicateEntities(RetrivalJob, ModificationJob):

    job: RetrivalJob

    @property
    def entity_columns(self) -> list[str]:
        return self.job.request_result.entity_columns

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()
        return df.unique(subset=self.entity_columns)

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_pandas()
        return df.drop_duplicates(subset=self.entity_columns)


@dataclass
class UpdateVectorIndexJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    indexes: list[VectorIndex]

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    async def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        data = await self.job.to_lazy_polars()

        update_jobs = []
        for index in self.indexes:

            select = index.entities + index.metadata + [index.vector]
            select_names = {feat.name for feat in select}

            filter_expr = pl.lit(True)
            for entity in index.entities:
                filter_expr = filter_expr & pl.col(entity.name).is_not_null()

            filtered_data = data.select(select_names).filter(filter_expr)
            update_jobs.append(index.storage.upsert_polars(filtered_data, index))

        await asyncio.gather(*update_jobs)
        return data


@dataclass
class LiteralDictJob(RetrivalJob):

    data: dict[str, list]
    requests: list[RetrivalRequest]

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.requests

    async def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return pl.DataFrame(self.data).lazy()

    def describe(self) -> str:
        return f'LiteralDictJob {self.data}'


@dataclass
class LogJob(RetrivalJob, ModificationJob):

    job: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    async def to_pandas(self) -> pd.DataFrame:
        if logger.level == 0:
            logging.basicConfig(level=logging.DEBUG)

        job_name = self.retrival_requests[0].name
        logging.debug(f'Starting to run {type(self.job).__name__} - {job_name}')
        try:
            df = await self.job.to_pandas()
        except Exception as error:
            logging.debug(f'Failed in job: {type(self.job).__name__} - {job_name}')
            raise error
        logging.debug(f'Results from {type(self.job).__name__} - {job_name}')
        logging.debug(df.columns)
        logging.debug(df)
        return df

    async def to_lazy_polars(self) -> pl.LazyFrame:
        if logger.level == 0:
            logging.basicConfig(level=logging.DEBUG)
        job_name = self.retrival_requests[0].name
        logging.debug(f'Starting to run {type(self.job).__name__} - {job_name}')
        try:
            df = await self.job.to_lazy_polars()
        except Exception as error:
            logging.debug(f'Failed in job: {type(self.job).__name__} - {job_name}')
            raise error
        logging.debug(f'Results from {type(self.job).__name__} - {job_name}')
        logging.debug(df.columns)
        logging.debug(df.head(10).collect())
        return df

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()

    def log_each_job(self) -> RetrivalJob:
        return self.job


@dataclass
class DropInvalidJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    validator: Validator

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    @staticmethod
    def features_to_validate(retrival_requests: list[RetrivalRequest]) -> set[Feature]:
        result = RequestResult.from_request_list(retrival_requests)
        return result.features.union(result.entities)

    async def to_pandas(self) -> pd.DataFrame:
        return self.validator.validate_pandas(
            list(DropInvalidJob.features_to_validate(self.retrival_requests)), await self.job.to_pandas()
        )

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return self.validator.validate_polars(
            list(DropInvalidJob.features_to_validate(self.retrival_requests)), await self.job.to_lazy_polars()
        )

    def with_subfeatures(self) -> RetrivalJob:
        return DropInvalidJob(self.job.with_subfeatures(), self.validator)

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        if isinstance(location, str):
            from aligned.sources.local import ParquetFileSource

            return FileCachedJob(ParquetFileSource(location), self)
        else:
            return FileCachedJob(location, self)

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()


@dataclass
class DerivedFeatureJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    requests: list[RetrivalRequest]

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    async def compute_derived_features_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:

        for request in self.requests:

            missing_features = request.features_to_include - set(df.columns)

            if len(missing_features) == 0:
                logger.debug('Skipping to compute derived features as they are already computed')
                continue

            for feature_round in request.derived_features_order():

                round_expressions: list[pl.Expr] = []

                for feature in feature_round:
                    if feature.transformation.should_skip(feature.name, df.columns):
                        logger.debug(f'Skipped adding feature {feature.name} to computation plan')
                        continue

                    logger.debug(f'Adding feature to computation plan in polars: {feature.name}')

                    method = await feature.transformation.transform_polars(df, feature.name)
                    if isinstance(method, pl.LazyFrame):
                        df = method
                    elif isinstance(method, pl.Expr):
                        round_expressions.append(method.alias(feature.name))
                    else:
                        raise ValueError('Invalid result from transformation')

                if round_expressions:
                    df = df.with_columns(round_expressions)
        return df

    async def compute_derived_features_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        for request in self.requests:
            for feature_round in request.derived_features_order():
                for feature in feature_round:
                    if feature.transformation.should_skip(feature.name, list(df.columns)):
                        logger.debug(f'Skipping to compute {feature.name} as it is aleady computed')
                        continue

                    logger.debug(f'Computing feature with pandas: {feature.name}')
                    df[feature.name] = await feature.transformation.transform_pandas(
                        df[feature.depending_on_names]
                    )
                    # if df[feature.name].dtype != feature.dtype.pandas_type:
                    #     if feature.dtype.is_numeric:
                    #         df[feature.name] = pd.to_numeric(df[feature.name], errors='coerce').astype(
                    #             feature.dtype.pandas_type
                    #         )
                    #     else:
                    #         df[feature.name] = df[feature.name].astype(feature.dtype.pandas_type)
        return df

    async def to_pandas(self) -> pd.DataFrame:
        return await self.compute_derived_features_pandas(await self.job.to_pandas())

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return await self.compute_derived_features_polars(await self.job.to_lazy_polars())

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()


@dataclass
class UniqueRowsJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    unique_on: list[str]  # type: ignore
    sort_key: str | None = field(default=None)

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        data = await self.job.to_lazy_polars()

        if self.sort_key:
            data = data.sort(self.sort_key, descending=True)

        return data.unique(self.unique_on, keep='first').lazy()


@dataclass
class ValidateEntitiesJob(RetrivalJob, ModificationJob):

    job: RetrivalJob

    async def to_pandas(self) -> pd.DataFrame:
        data = await self.job.to_pandas()

        for request in self.retrival_requests:
            if request.entity_names - set(data.columns):
                return pd.DataFrame({})

        return data

    async def to_lazy_polars(self) -> pl.LazyFrame:
        data = await self.job.to_lazy_polars()

        for request in self.retrival_requests:
            if request.entity_names - set(data.columns):
                return pl.DataFrame({}).lazy()

        return data


@dataclass
class FillMissingColumnsJob(RetrivalJob, ModificationJob):

    job: RetrivalJob

    async def to_pandas(self) -> pd.DataFrame:
        from aligned.schemas.constraints import Optional

        data = await self.job.to_pandas()
        for request in self.retrival_requests:

            optional_constraint = Optional()
            for feature in request.features:
                if (
                    feature.constraints
                    and optional_constraint in feature.constraints
                    and feature.name not in data.columns
                ):
                    data[feature] = None

        return data

    async def to_lazy_polars(self) -> pl.LazyFrame:
        from aligned.schemas.constraints import Optional

        data = await self.job.to_lazy_polars()
        optional_constraint = Optional()

        for request in self.retrival_requests:

            missing_columns = [
                feature.name
                for feature in request.features
                if feature.constraints
                and optional_constraint in feature.constraints
                and feature.name not in data.columns
            ]

            if missing_columns:
                data = data.with_columns([pl.lit(None).alias(feature) for feature in missing_columns])

        return data


@dataclass
class StreamAggregationJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    checkpoints: dict[AggregateOver, DataFileReference]

    @property
    def time_windows(self) -> set[AggregateOver]:
        windows = set()
        for request in self.retrival_requests:
            for feature in request.aggregated_features:
                windows.add(feature.aggregate_over)
        return windows

    @property
    def aggregated_features(self) -> dict[AggregateOver, set[AggregatedFeature]]:
        features = defaultdict(set)
        for request in self.retrival_requests:
            for feature in request.aggregated_features:
                features[feature.aggregate_over].add(feature)
        return features

    async def data_windows(self, window: AggregateOver, data: pl.DataFrame, now: datetime) -> pl.DataFrame:
        checkpoint = self.checkpoints[window]
        filter_expr: pl.Expr | None = None

        if window.window:
            time_window = window.window
            filter_expr = pl.col(time_window.time_column.name) > now - time_window.time_window

        if window.condition:
            raise ValueError('Condition is not supported for stream aggregation, yet')

        try:
            window_data = (await checkpoint.to_lazy_polars()).collect()

            if filter_expr is not None:
                new_data = pl.concat(
                    [window_data.filter(filter_expr), data.filter(filter_expr)], how='vertical_relaxed'
                )
            else:
                new_data = pl.concat([window_data, data], how='vertical_relaxed')

            await checkpoint.write_polars(new_data.lazy())
            return new_data
        except FileNotFoundError:

            if filter_expr is not None:
                window_data = data.filter(filter_expr)
            else:
                window_data = data

            await checkpoint.write_polars(window_data.lazy())
            return window_data

    async def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        data = (await self.job.to_lazy_polars()).collect()

        # This is used as a dummy frame, as the pl abstraction is not good enough
        lazy_df = pl.DataFrame({}).lazy()
        now = datetime.utcnow()

        for window in self.time_windows:

            aggregations = self.aggregated_features[window]

            required_features = set(window.group_by).union([window.window.time_column])
            for agg in aggregations:
                required_features.update(agg.derived_feature.depending_on)

            required_features_name = sorted({feature.name for feature in required_features})

            agg_transformations = await asyncio.gather(
                *[
                    agg.derived_feature.transformation.transform_polars(lazy_df, 'dummy')
                    for agg in aggregations
                ]
            )
            agg_expr = [
                agg.alias(feature.name)
                for agg, feature in zip(agg_transformations, aggregations)
                if isinstance(agg, pl.Expr)
            ]

            window_data = await self.data_windows(window, data.select(required_features_name), now)

            agg_data = window_data.lazy().groupby(window.group_by_names).agg(agg_expr).collect()
            data = data.join(agg_data, on=window.group_by_names, how='left')

        return data.lazy()

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()


@dataclass
class DataLoaderJob:

    job: RetrivalJob
    chunk_size: int

    async def to_polars(self) -> AsyncIterator[pl.LazyFrame]:
        from math import ceil

        from aligned.local.job import LiteralRetrivalJob

        needed_requests = self.job.retrival_requests
        without_derived = self.job.remove_derived_features()
        raw_files = (await without_derived.to_lazy_polars()).collect()
        features_to_include = self.job.request_result.features.union(self.job.request_result.entities)
        features_to_include_names = {feature.name for feature in features_to_include}

        iterations = ceil(raw_files.shape[0] / self.chunk_size)
        for i in range(iterations):
            start = i * self.chunk_size
            end = (i + 1) * self.chunk_size
            df = raw_files[start:end, :]

            chunked_job = (
                LiteralRetrivalJob(df.lazy(), RequestResult.from_request_list(needed_requests))
                .derive_features(needed_requests)
                .select_columns(features_to_include_names)
            )

            chunked_df = await chunked_job.to_lazy_polars()
            yield chunked_df

    async def to_pandas(self) -> AsyncIterator[pd.DataFrame]:
        async for chunk in self.to_polars():
            yield chunk.collect().to_pandas()


@dataclass
class RawFileCachedJob(RetrivalJob, ModificationJob):

    location: DataFileReference
    job: DerivedFeatureJob

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    async def to_pandas(self) -> pd.DataFrame:
        from aligned.local.job import FileFullJob
        from aligned.sources.local import LiteralReference

        try:
            logger.debug('Trying to read cache file')
            df = await self.location.read_pandas()
        except UnableToFindFileException:
            logger.debug('Unable to load file, so fetching from source')
            df = await self.job.job.to_pandas()
            logger.debug('Writing result to cache')
            await self.location.write_pandas(df)
        return (
            await FileFullJob(LiteralReference(df), request=self.job.requests[0])
            .derive_features(self.job.requests)
            .to_pandas()
        )

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return await self.job.to_lazy_polars()

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        return self

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()


@dataclass
class FileCachedJob(RetrivalJob, ModificationJob):

    location: DataFileReference
    job: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    async def to_pandas(self) -> pd.DataFrame:
        try:
            logger.debug('Trying to read cache file')
            df = await self.location.read_pandas()
        except UnableToFindFileException:
            logger.debug('Unable to load file, so fetching from source')
            df = await self.job.to_pandas()
            logger.debug('Writing result to cache')
            await self.location.write_pandas(df)
        return df

    async def to_lazy_polars(self) -> pl.LazyFrame:
        try:
            logger.debug('Trying to read cache file')
            df = await self.location.to_lazy_polars()
        except UnableToFindFileException:
            logger.debug('Unable to load file, so fetching from source')
            df = await self.job.to_lazy_polars()
            logger.debug('Writing result to cache')
            await self.location.write_polars(df)
        except FileNotFoundError:
            logger.debug('Unable to load file, so fetching from source')
            df = await self.job.to_lazy_polars()
            logger.debug('Writing result to cache')
            await self.location.write_polars(df)
        return df

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        return self

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()


@dataclass
class SplitJob:

    job: RetrivalJob
    target_column: str
    strategy: SplitStrategy

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    async def use_pandas(self) -> SplitDataSet[pd.DataFrame]:
        data = await self.job.to_pandas()
        return self.strategy.split_pandas(data, self.target_column)

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()


@dataclass
class WithRequests(RetrivalJob, ModificationJob):

    job: RetrivalJob
    requests: list[RetrivalRequest]

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.requests

    async def to_pandas(self) -> pd.DataFrame:
        return await self.job.to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return await self.job.to_lazy_polars()


@dataclass
class TimeMetricLoggerJob(RetrivalJob, ModificationJob):

    job: RetrivalJob

    time_metric: Histogram
    labels: list[str] | None = field(default=None)

    async def to_pandas(self) -> pd.DataFrame:
        start_time = timeit.default_timer()
        df = await self.job.to_pandas()
        elapsed = timeit.default_timer() - start_time
        logger.debug(f'Computed records in {elapsed} seconds')
        if self.labels:
            self.time_metric.labels(*self.labels).observe(elapsed)
        else:
            self.time_metric.observe(elapsed)
        return df

    async def to_lazy_polars(self) -> pl.LazyFrame:
        start_time = timeit.default_timer()
        df = await self.job.to_lazy_polars()
        concrete = df.collect()
        elapsed = timeit.default_timer() - start_time
        logger.debug(f'Computed records in {elapsed} seconds')
        if self.labels:
            self.time_metric.labels(*self.labels).observe(elapsed)
        else:
            self.time_metric.observe(elapsed)
        return concrete.lazy()


@dataclass
class EnsureTypesJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    requests: list[RetrivalRequest]
    date_formatter: DateFormatter = field(default_factory=DateFormatter.iso_8601)

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.requests

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_pandas()
        for request in self.requests:
            features_to_check = request.all_required_features

            if request.aggregated_features:
                features_to_check = {feature.derived_feature for feature in request.aggregated_features}

            for feature in features_to_check:

                mask = ~df[feature.name].isnull()

                with suppress(AttributeError, TypeError):
                    df[feature.name] = df[feature.name].mask(
                        ~mask, other=df.loc[mask, feature.name].str.strip('"')
                    )

                if feature.dtype == FeatureType.datetime():
                    df[feature.name] = pd.to_datetime(df[feature.name], infer_datetime_format=True, utc=True)
                elif feature.dtype == FeatureType.datetime() or feature.dtype == FeatureType.string():
                    continue
                elif (feature.dtype == FeatureType.array()) or (feature.dtype == FeatureType.embedding()):
                    import json

                    if df[feature.name].dtype == 'object':
                        df[feature.name] = df[feature.name].apply(
                            lambda x: json.loads(x) if isinstance(x, str) else x
                        )
                elif feature.dtype == FeatureType.json():
                    pass
                elif feature.dtype == FeatureType.datetime():
                    pass
                else:
                    if feature.dtype.is_numeric:
                        df[feature.name] = pd.to_numeric(df[feature.name], errors='coerce').astype(
                            feature.dtype.pandas_type
                        )
                    else:
                        df[feature.name] = df[feature.name].astype(feature.dtype.pandas_type)

            if request.event_timestamp and request.event_timestamp.name in df.columns:
                feature = request.event_timestamp
                df[feature.name] = pd.to_datetime(df[feature.name], infer_datetime_format=True, utc=True)
        return df

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()
        org_schema = dict(df.schema)
        for request in self.requests:
            features_to_check = request.all_required_features

            if request.aggregated_features:
                features_to_check.update({feature.derived_feature for feature in request.aggregated_features})

            for feature in features_to_check:

                if feature.name not in org_schema:
                    continue

                if feature.dtype.polars_type.is_(org_schema[feature.name]):
                    continue

                if feature.dtype == FeatureType.bool():
                    df = df.with_columns(pl.col(feature.name).cast(pl.Int8).cast(pl.Boolean))
                elif feature.dtype == FeatureType.datetime():
                    current_dtype = df.select([feature.name]).dtypes[0]

                    if isinstance(current_dtype, pl.Datetime):
                        continue

                    df = df.with_columns(self.date_formatter.decode_polars(feature.name))
                elif (feature.dtype == FeatureType.array()) or (feature.dtype == FeatureType.embedding()):
                    dtype = df.select(feature.name).dtypes[0]
                    if dtype == pl.Utf8:
                        df = df.with_columns(pl.col(feature.name).str.json_extract(pl.List(pl.Utf8)))
                elif feature.dtype == FeatureType.json():
                    pass
                elif feature.dtype == FeatureType.datetime():
                    pass
                else:
                    df = df.with_columns(pl.col(feature.name).cast(feature.dtype.polars_type, strict=False))

            if request.event_timestamp:
                feature = request.event_timestamp
                if feature.name not in df.columns:
                    continue
                current_dtype = df.select([feature.name]).dtypes[0]

                if not isinstance(current_dtype, pl.Datetime):
                    df = df.with_columns(
                        (pl.col(feature.name).cast(pl.Int64) * 1000)
                        .cast(pl.Datetime(time_zone='UTC'))
                        .alias(feature.name)
                    )

        return df

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()


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

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        jobs = []
        for job in self.jobs:
            jobs.extend(job.retrival_requests)
        return jobs + self.combined_requests

    def ignore_event_timestamp(self) -> RetrivalJob:
        return CombineFactualJob([job.ignore_event_timestamp() for job in self.jobs], self.combined_requests)

    async def combine_data(self, df: pd.DataFrame) -> pd.DataFrame:
        for request in self.combined_requests:
            for feature in request.derived_features:
                if feature.name in df.columns:
                    logger.debug(f'Skipping feature {feature.name}, already computed')
                    continue
                logger.debug(f'Computing feature: {feature.name}')
                df[feature.name] = await feature.transformation.transform_pandas(
                    df[feature.depending_on_names]
                )
        return df

    async def combine_polars_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        for request in self.combined_requests:
            logger.debug(f'{request.name}, {len(request.derived_features)}')
            for feature in request.derived_features:
                if feature.name in df.columns:
                    logger.debug(f'Skipping feature {feature.name}, already computed')
                    continue
                logger.debug(f'Computing feature: {feature.name}')
                result = await feature.transformation.transform_polars(df, feature.name)
                if isinstance(result, pl.Expr):
                    df = df.with_columns([result.alias(feature.name)])
                elif isinstance(result, pl.LazyFrame):
                    df = result
                else:
                    raise ValueError(f'Unsupported transformation result type, got {type(result)}')
        return df

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        if not self.jobs:
            raise ValueError(
                'Have no jobs to fetch. This is probably an internal error.\n'
                'Please submit an issue, and describe how to reproduce it.\n'
                'Or maybe even submit a PR'
            )

        dfs: list[pl.LazyFrame] = await asyncio.gather(*[job.to_lazy_polars() for job in self.jobs])
        results = [job.request_result for job in self.jobs]

        joined_entities = set(results[0].entity_columns)
        df = dfs[0].collect()

        for other_df, job_result in list(zip(dfs, results))[1:]:

            other_df = other_df.collect()

            join_on = job_result.entity_columns

            if df.height == other_df.height:
                df = df.lazy().with_context(other_df.lazy()).select(pl.all()).collect()
            elif df.height > other_df.height:
                df = df.join(other_df, on=join_on)
            else:
                df = other_df.join(df, on=list(joined_entities))

            joined_entities.update(job_result.entity_columns)

        return await self.combine_polars_data(df.lazy())

    def cache_raw_data(self, location: DataFileReference | str) -> RetrivalJob:
        return CombineFactualJob([job.cache_raw_data(location) for job in self.jobs], self.combined_requests)

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        return CombineFactualJob([job.cached_at(location) for job in self.jobs], self.combined_requests)

    def remove_derived_features(self) -> RetrivalJob:
        return CombineFactualJob([job.remove_derived_features() for job in self.jobs], self.combined_requests)

    def log_each_job(self) -> RetrivalJob:
        return LogJob(CombineFactualJob([job.log_each_job() for job in self.jobs], self.combined_requests))

    def describe(self) -> str:
        description = f'Combining {len(self.jobs)} jobs:\n'
        for index, job in enumerate(self.jobs):
            description += f'{index + 1}: {job.describe()}\n'
        return description


@dataclass
class SelectColumnsJob(RetrivalJob, ModificationJob):

    include_features: set[str]
    job: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result.filter_features(self.include_features)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return [request.filter_features(self.include_features) for request in self.job.retrival_requests]

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_pandas()
        if self.include_features:
            total_list = list({ent.name for ent in self.request_result.entities}.union(self.include_features))
            return df[total_list]
        else:
            return df

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()
        if self.include_features:
            total_list = list({ent.name for ent in self.request_result.entities}.union(self.include_features))
            return df.select(total_list)
        else:
            return df

    def drop_invalid(self, validator: Validator) -> RetrivalJob:
        return SelectColumnsJob(self.include_features, self.job.drop_invalid(validator))

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        return SelectColumnsJob(self.include_features, self.job.cached_at(location))

    def with_subfeatures(self) -> RetrivalJob:
        return self.job.with_subfeatures()

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()

    def ignore_event_timestamp(self) -> RetrivalJob:
        return SelectColumnsJob(
            include_features=self.include_features - {'event_timestamp'},
            job=self.job.ignore_event_timestamp(),
        )


@dataclass
class ListenForTriggers(RetrivalJob, ModificationJob):

    job: RetrivalJob
    triggers: set[EventTrigger]

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    async def to_pandas(self) -> pd.DataFrame:
        import asyncio

        df = await self.job.to_pandas()
        await asyncio.gather(*[trigger.check_pandas(df, self.request_result) for trigger in self.triggers])
        return df

    async def to_lazy_polars(self) -> pl.LazyFrame:
        import asyncio

        df = await self.job.to_lazy_polars()
        await asyncio.gather(*[trigger.check_polars(df, self.request_result) for trigger in self.triggers])
        return df

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()


@dataclass
class CustomLazyPolarsJob(RetrivalJob):

    request: RetrivalRequest
    method: Callable[[], Coroutine[None, None, pl.LazyFrame]]

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return [self.request]

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request(self.request)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return await self.method()

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_polars()).to_pandas()
