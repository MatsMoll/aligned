from __future__ import annotations
from io import StringIO
from aligned.schemas.date_formatter import DateFormatter

from pytz import timezone
import asyncio
import logging
import timeit
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Collection, Literal, Union, TypeVar, Coroutine, Any

import polars as pl
from aligned.lazy_imports import pandas as pd

from polars.type_aliases import TimeUnit
from prometheus_client import Histogram

from aligned.exceptions import UnableToFindFileException

from aligned.request.retrival_request import FeatureRequest, RequestResult, RetrivalRequest
from aligned.schemas.feature import Feature, FeatureLocation, FeatureType
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.vector_storage import VectorIndex
from aligned.split_strategy import SupervisedDataSet
from aligned.validation.interface import Validator, PolarsValidator

if TYPE_CHECKING:
    from typing import AsyncIterator
    from aligned.sources.local import Directory
    from aligned.schemas.folder import DatasetMetadata, DatasetStore
    from aligned.feature_source import WritableFeatureSource

    from aligned.schemas.derivied_feature import AggregatedFeature, AggregateOver
    from aligned.schemas.model import EventTrigger, Model
    from aligned.sources.local import DataFileReference, StorageFileReference
    from aligned.feature_store import ContractStore


logger = logging.getLogger(__name__)


def split(
    data: pd.DataFrame, start_ratio: float, end_ratio: float, event_timestamp_column: str | None = None
) -> pd.Index:
    index = pd.Index([], dtype=data.index.dtype)
    if event_timestamp_column:
        column = data[event_timestamp_column]
        if column.dtype != 'datetime64[ns]':
            column = pd.to_datetime(data[event_timestamp_column])
        data = data.iloc[column.sort_values().index]  # type: ignore

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


def fraction_from_job(job: RetrivalJob) -> float | None:
    if isinstance(job, SubsetJob):
        return job.fraction
    elif isinstance(job, InMemorySplitCacheJob):
        return job.dataset_sizes[job.dataset_index]
    return None


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

    async def store_dataset_at_directory(
        self,
        directory: Directory,
        dataset_store: DatasetStore | StorageFileReference,
        metadata: DatasetMetadata | None = None,
        id: str | None = None,
        tags: list[str] | None = None,
    ) -> TrainTestJob:
        from uuid import uuid4
        from aligned.schemas.folder import DatasetMetadata

        if not metadata:
            metadata = DatasetMetadata(
                id=id or str(uuid4()),
                name='train_test - ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                description='A train and test dataset.',
                tags=tags,
            )

        run_dir = directory.sub_directory(metadata.id)
        return await self.store_dataset(
            dataset_store=dataset_store,
            train_source=run_dir.parquet_at('train.parquet'),  # type: ignore
            test_source=run_dir.parquet_at('test.parquet'),  # type: ignore
            metadata=metadata,
        )

    async def store_dataset(
        self,
        dataset_store: DatasetStore | StorageFileReference,
        metadata: DatasetMetadata,
        train_source: DataFileReference,
        test_source: DataFileReference,
        train_size: float | None = None,
        test_size: float | None = None,
    ) -> TrainTestJob:
        from aligned.schemas.folder import (
            TrainDatasetMetadata,
            JsonDatasetStore,
            StorageFileSource,
            DatasetStore,
        )
        from aligned.data_source.batch_data_source import CodableBatchDataSource

        request_result = self.train_job.request_result

        if isinstance(dataset_store, StorageFileSource):
            data_store = JsonDatasetStore(dataset_store)
        elif isinstance(dataset_store, DatasetStore):
            data_store = dataset_store
        else:
            raise ValueError(f'Unknown dataset store type: {type(dataset_store)}')

        if train_size is None:
            train_size = fraction_from_job(self.train_job)

        if test_size is None:
            test_size = fraction_from_job(self.test_job)

        if not isinstance(test_source, CodableBatchDataSource):
            raise ValueError('test_source should be a BatchDataSource')

        if not isinstance(train_source, CodableBatchDataSource):
            raise ValueError('train_source should be a BatchDataSource')

        test_metadata = TrainDatasetMetadata(
            id=metadata.id,
            name=metadata.name,
            content=request_result,
            description=metadata.description,
            train_size_fraction=train_size,
            test_size_fraction=test_size,
            train_dataset=train_source,
            test_dataset=test_source,
            target=list(self.target_columns),
        )

        async def update_metadata() -> None:
            await data_store.store_train_test(test_metadata)

        _ = await (
            self.train_job.cached_at(train_source)
            .on_load(update_metadata)
            .cached_at(train_source)
            .to_lazy_polars()
        )
        _ = await (
            self.test_job.cached_at(test_source)
            .on_load(update_metadata)
            .cached_at(test_source)
            .to_lazy_polars()
        )

        return TrainTestJob(
            train_job=train_source.all(request_result),
            test_job=test_source.all(request_result),
            target_columns=self.target_columns,
        )


@dataclass
class TrainTestValidateJob:

    train_job: RetrivalJob
    test_job: RetrivalJob
    validate_job: RetrivalJob

    target_columns: set[str]

    should_filter_out_null_targets: bool = True

    @property
    def train(self) -> SupervisedJob:
        return SupervisedJob(self.train_job, self.target_columns, self.should_filter_out_null_targets)

    @property
    def test(self) -> SupervisedJob:
        return SupervisedJob(self.test_job, self.target_columns, self.should_filter_out_null_targets)

    @property
    def validate(self) -> SupervisedJob:
        return SupervisedJob(self.validate_job, self.target_columns, self.should_filter_out_null_targets)

    async def store_dataset_at_directory(
        self,
        directory: Directory,
        dataset_store: DatasetStore | StorageFileReference,
        metadata: DatasetMetadata | None = None,
        id: str | None = None,
        tags: list[str] | None = None,
    ) -> TrainTestValidateJob:
        from uuid import uuid4
        from aligned.schemas.folder import DatasetMetadata

        if not metadata:
            metadata = DatasetMetadata(
                id=id or str(uuid4()),
                name='train_test_validate - ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                description='A train, test and validation dataset.',
                tags=tags,
            )

        run_dir = directory.sub_directory(metadata.id)
        return await self.store_dataset(
            dataset_store=dataset_store,
            train_source=run_dir.parquet_at('train.parquet'),  # type: ignore
            test_source=run_dir.parquet_at('test.parquet'),  # type: ignore
            validate_source=run_dir.parquet_at('validate.parquet'),  # type: ignore
            metadata=metadata,
        )

    async def store_dataset(
        self,
        dataset_store: DatasetStore | StorageFileReference,
        train_source: DataFileReference,
        test_source: DataFileReference,
        validate_source: DataFileReference,
        metadata: DatasetMetadata | None = None,
        train_size: float | None = None,
        test_size: float | None = None,
        validation_size: float | None = None,
    ) -> TrainTestValidateJob:
        from aligned.schemas.folder import (
            TrainDatasetMetadata,
            JsonDatasetStore,
            DatasetMetadata,
            DatasetStore,
        )
        from aligned.data_source.batch_data_source import CodableBatchDataSource
        from aligned.sources.local import StorageFileSource
        from uuid import uuid4

        if metadata is None:
            metadata = DatasetMetadata(
                id=str(uuid4()),
                name='train_test_validate - ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                description='A train, test and validation dataset.',
            )

        if isinstance(dataset_store, StorageFileSource):
            data_store = JsonDatasetStore(dataset_store)
        elif isinstance(dataset_store, DatasetStore):
            data_store = dataset_store
        else:
            raise ValueError(f'Unknown dataset store type: {type(dataset_store)}')

        request_result = self.train_job.request_result

        if train_size is None:
            train_size = fraction_from_job(self.train_job)

        if test_size is None:
            test_size = fraction_from_job(self.test_job)

        if validation_size is None:
            validation_size = fraction_from_job(self.validate_job)

        if not isinstance(test_source, CodableBatchDataSource):
            raise ValueError('test_source should be a BatchDataSource')

        if not isinstance(train_source, CodableBatchDataSource):
            raise ValueError('train_source should be a BatchDataSource')

        if not isinstance(validate_source, CodableBatchDataSource):
            raise ValueError('validation_source should be a BatchDataSource')

        test_metadata = TrainDatasetMetadata(
            id=metadata.id,
            name=metadata.name,
            content=request_result,
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

        _ = (
            await self.train_job.cached_at(train_source)
            .on_load(update_metadata)
            .cached_at(train_source)
            .to_lazy_polars()
        )
        _ = (
            await self.test_job.cached_at(test_source)
            .on_load(update_metadata)
            .cached_at(test_source)
            .to_lazy_polars()
        )
        _ = await (
            self.validate_job.cached_at(validate_source).on_load(update_metadata).cached_at(validate_source)
        ).to_lazy_polars()

        return TrainTestValidateJob(
            train_job=train_source.all(request_result),
            test_job=test_source.all(request_result),
            validate_job=validate_source.all(request_result),
            target_columns=self.target_columns,
        )


SplitterCallable = Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]


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
            data = data.drop_nulls(list(self.target_columns))

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

    def train_test(
        self, train_size: float, splitter_factory: Callable[[SplitConfig], SplitterCallable] | None = None
    ) -> TrainTestJob:

        cached_job = InMemoryCacheJob(self.job)

        event_timestamp = self.job.request_result.event_timestamp

        train_config = SplitConfig(
            left_size=train_size,
            right_size=1 - train_size,
            event_timestamp_column=event_timestamp,
            target_columns=list(self.target_columns),
        )

        if splitter_factory:
            train_splitter = splitter_factory(train_config)  # type: ignore
        else:

            def train_splitter(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
                return (
                    subset_polars(df, 0, train_config.left_size, event_timestamp),
                    subset_polars(df, train_config.left_size, 1, event_timestamp),
                )

        train_job, test_job = cached_job.split(train_splitter, (train_size, 1 - train_size))
        return TrainTestJob(
            train_job=train_job,
            test_job=test_job,
            target_columns=self.target_columns,
        )

    def train_test_validate(
        self,
        train_size: float,
        validate_size: float,
        splitter_factory: Callable[[SplitConfig], SplitterCallable] | None = None,
    ) -> TrainTestValidateJob:

        job_to_cache = self.job
        if self.should_filter_out_null_targets:
            job_to_cache = self.job.polars_method(lambda df: df.drop_nulls(self.target_columns))

        event_timestamp = self.job.request_result.event_timestamp

        leftover_size = 1 - train_size

        train_config = SplitConfig(
            left_size=train_size,
            right_size=leftover_size,
            event_timestamp_column=event_timestamp,
            target_columns=list(self.target_columns),
        )
        test_config = SplitConfig(
            left_size=(leftover_size - validate_size) / leftover_size,
            right_size=validate_size / leftover_size,
            event_timestamp_column=event_timestamp,
            target_columns=list(self.target_columns),
        )

        if splitter_factory:
            train_splitter = splitter_factory(train_config)  # type: ignore
            validate_splitter = splitter_factory(test_config)  # type: ignore
        else:

            def train_splitter(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
                return (
                    subset_polars(df, 0, train_config.left_size, event_timestamp),
                    subset_polars(df, train_config.left_size, 1, event_timestamp),
                )

            def validate_splitter(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
                return (
                    subset_polars(df, 0, test_config.left_size, event_timestamp),
                    subset_polars(df, test_config.left_size, 1, event_timestamp),
                )

        train_job, rem_job = job_to_cache.split(train_splitter, (train_size, 1 - train_size))
        test_job, validate_job = rem_job.split(
            validate_splitter, (1 - train_size - validate_size, validate_size)
        )

        return TrainTestValidateJob(
            train_job=train_job,
            test_job=test_job,
            validate_job=validate_job,
            target_columns=self.target_columns,
            should_filter_out_null_targets=self.should_filter_out_null_targets,
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

    def drop_invalid(self, validator: Validator | None = None) -> SupervisedJob:
        return SupervisedJob(
            self.job.drop_invalid(validator),
            self.target_columns,
        )

    def log_each_job(self, logger_func: Callable[[object], None] | None = None) -> SupervisedJob:
        return SupervisedJob(
            self.job.log_each_job(logger_func),
            self.target_columns,
        )

    def unpack_embeddings(self) -> SupervisedJob:
        return SupervisedJob(
            self.job.unpack_embeddings(),
            self.target_columns,
        )

    def describe(self) -> str:
        return f'{self.job.describe()} with target columns {self.target_columns}'


ConvertableToRetrivalJob = Union[
    list[dict[str, Any]], dict[str, list], 'pd.DataFrame', pl.DataFrame, pl.LazyFrame
]


class RetrivalJob(ABC):
    @property
    def loaded_columns(self) -> list[str]:
        if isinstance(self, ModificationJob):
            return self.job.loaded_columns
        return []

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

    def inject_store(self, store: ContractStore) -> RetrivalJob:
        if isinstance(self, ModificationJob):
            return self.copy_with(self.job.inject_store(store))
        return self

    def describe(self) -> str:
        if isinstance(self, ModificationJob):
            return f'{self.job.describe()} -> {self.__class__.__name__}'
        raise NotImplementedError(f'Describe not implemented for {self.__class__.__name__}')

    def remove_derived_features(self) -> RetrivalJob:
        return self.without_derived_features()

    def without_derived_features(self) -> RetrivalJob:
        if isinstance(self, ModificationJob):
            return self.copy_with(self.job.remove_derived_features())
        return self

    def log_each_job(self, logger_func: Callable[[object], None] | None = None) -> RetrivalJob:
        if isinstance(self, ModificationJob):
            return LogJob(self.copy_with(self.job.log_each_job(logger_func)))
        return LogJob(self, logger_func or logger.debug)

    def unpack_embeddings(self) -> RetrivalJob:
        if isinstance(self, ModificationJob):
            return self.copy_with(self.job.unpack_embeddings())
        return self

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

    def return_invalid(self, should_return_validation: bool | None = None) -> RetrivalJob:
        if should_return_validation is None:
            should_return_validation = False
        return ReturnInvalidJob(self, should_return_validation)

    def split(
        self,
        splitter: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]],
        dataset_sizes: tuple[float, float],
    ) -> tuple[RetrivalJob, RetrivalJob]:

        job = InMemorySplitCacheJob(self, splitter, dataset_sizes, 0)
        return (job, job.with_dataset_index(1))

    def join(
        self,
        job: RetrivalJob,
        method: Literal['inner', 'left', 'outer'],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> RetrivalJob:

        if isinstance(left_on, str):
            left_on = [left_on]

        if isinstance(right_on, str):
            right_on = [right_on]

        return JoinJobs(method=method, left_job=self, right_job=job, left_on=left_on, right_on=right_on)

    def on_load(self, on_load: Callable[[], Coroutine[Any, Any, None]]) -> RetrivalJob:
        return OnLoadJob(self, on_load)

    def filter(self, condition: str | Feature | DerivedFeature | pl.Expr) -> RetrivalJob:
        """
        Filters based on a condition referencing either a feature,
        a feature name, or an polars expression to filter on.
        """
        if isinstance(self, ModificationJob):
            return self.copy_with(self.job.filter(condition))
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

    def drop_invalid(self, validator: Validator | None = None) -> RetrivalJob:
        """
        Drops invalid row based on the defined features.

        ```python
        @feature_view(...)
        class WhiteWine:
            wine_id = UInt64().as_entity()

            quality = Int32().lower_bound(1).upper_bound(10)


        valid_wines = WhiteWine.drop_invalid({
            "wine_id": [0, 1, 2, 3, 4],
            "quality": [None, 4, 8, 20, -10]
        })

        print(valid_wines)
        >>> {
            "wine_id": [1, 2],
            "quality": [4, 8]
        }
        ```
        Args:
            validator (Validator): A validator that can validate the data.
                The default uses the `PolarsValidator`

        Returns:
            RetrivalJob: A new retrival job with only valid rows.
        """
        if isinstance(self, ModificationJob):
            return self.copy_with(self.job.drop_invalid(validator))
        return DropInvalidJob(self, validator or PolarsValidator())

    def monitor_time_used(self, time_metric: Histogram, labels: list[str] | None = None) -> RetrivalJob:
        return TimeMetricLoggerJob(self, time_metric, labels)

    def derive_features(self, requests: list[RetrivalRequest] | None = None) -> RetrivalJob:
        requests = requests or self.retrival_requests

        for request in requests:
            if request.derived_features:
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

    def select(self, include_features: Collection[str]) -> RetrivalJob:
        return SelectColumnsJob(list(include_features), self)

    def select_columns(self, include_features: Collection[str]) -> RetrivalJob:
        return self.select(include_features)

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

    def transform_polars(self, polars_method: CustomPolarsTransform) -> RetrivalJob:
        return CustomPolarsJob(self, polars_method)

    def polars_method(self, polars_method: Callable[[pl.LazyFrame], pl.LazyFrame]) -> RetrivalJob:
        return self.transform_polars(polars_method)

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
    def from_lazy_function(
        callable: Callable[[], Coroutine[None, None, pl.LazyFrame]], request: RetrivalRequest
    ) -> RetrivalJob:
        return CustomLazyPolarsJob(request=request, method=callable)

    @staticmethod
    def from_convertable(
        data: ConvertableToRetrivalJob, request: list[RetrivalRequest] | RetrivalRequest | FeatureRequest
    ) -> RetrivalJob:
        import polars as pl
        from aligned.local.job import LiteralRetrivalJob

        if isinstance(request, RetrivalRequest):
            request = [request]
        elif isinstance(request, FeatureRequest):
            request = request.needed_requests

        def remove_features(
            loaded_features: set[str], requests: list[RetrivalRequest]
        ) -> list[RetrivalRequest]:
            revised_requests: list[RetrivalRequest] = []
            for req in requests:
                revised_requests.append(
                    RetrivalRequest(
                        name=req.name,
                        location=req.location,
                        entities=req.entities,
                        features={feat for feat in req.features if feat.name in loaded_features},
                        derived_features=req.derived_features,
                        event_timestamp_request=req.event_timestamp_request,
                    )
                )

            return revised_requests

        def add_additional_features(
            schema: dict[str, pl.DataType], requests: list[RetrivalRequest]
        ) -> list[RetrivalRequest]:

            additional_features = set()

            for req in requests:
                req_feature_names = req.all_returned_columns
                additional_features.update(
                    {
                        Feature(feat, FeatureType.from_polars(dtype))
                        for feat, dtype in schema.items()
                        if feat not in req_feature_names
                    }
                )

            if additional_features:
                return requests + [
                    RetrivalRequest(
                        name='additional',
                        location=FeatureLocation.feature_view('additional'),
                        entities=set(),
                        features=additional_features,
                        derived_features=set(),
                    )
                ]
            return requests

        loaded_features: set[str] = set()

        if isinstance(data, dict):
            loaded_features.update(data.keys())
        elif isinstance(data, list):
            assert isinstance(data[0], dict)
            loaded_features.update(data[0].keys())
        elif isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            loaded_features.update(data.columns)
        elif isinstance(data, pd.DataFrame):
            loaded_features.update(data.columns)

        schema: dict[str, pl.DataType] = {}
        requests = remove_features(loaded_features, request)
        for req in requests:
            feature_names = req.feature_names
            for feat, dtype in req.polars_schema().items():
                if feat in feature_names:
                    schema[feat] = dtype

        if isinstance(data, dict):
            df = pl.DataFrame(data, schema_overrides=schema).lazy()
        elif isinstance(data, list):
            df = pl.DataFrame(data, schema_overrides=schema).lazy()
        elif isinstance(data, pl.DataFrame):
            df = data.cast(schema).lazy()  # type: ignore
        elif isinstance(data, pl.LazyFrame):
            df = data.cast(schema)  # type: ignore
        elif isinstance(data, pd.DataFrame):
            df = pl.from_pandas(data, schema_overrides=schema).lazy()
        else:
            raise ValueError(f'Unable to convert {type(data)} to RetrivalJob')

        return LiteralRetrivalJob(df, add_additional_features(df.schema, requests))

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
            requests = self.retrival_requests
            if len(requests) > 1:
                request = RetrivalRequest.unsafe_combine(requests)
            else:
                assert len(requests) == 1, 'No requests. this should not happen and is a but'
                request = requests[0]
            await source.insert(self, request)


JobType = TypeVar('JobType')


class ModificationJob:

    job: RetrivalJob

    def copy_with(self: JobType, job: RetrivalJob) -> JobType:
        self.job = job  # type: ignore
        return self


def polars_filter_expressions_from(features: list[Feature]) -> list[tuple[pl.Expr, str]]:
    from aligned.schemas.constraints import (
        Optional,
        LowerBound,
        UpperBound,
        InDomain,
        MinLength,
        MaxLength,
        EndsWith,
        StartsWith,
        LowerBoundInclusive,
        UpperBoundInclusive,
        Regex,
    )

    optional_constraint = Optional()

    exprs: list[tuple[pl.Expr, str]] = []

    for feature in features:
        if not feature.constraints:
            exprs.append((pl.col(feature.name).is_not_null(), f"Required {feature.name}"))
            continue

        if optional_constraint not in feature.constraints:
            exprs.append((pl.col(feature.name).is_not_null(), f"Required {feature.name}"))
            continue

        for constraint in feature.constraints:
            if isinstance(constraint, LowerBound):
                exprs.append(
                    (pl.col(feature.name) > constraint.value, f"LowerBound {feature.name} {constraint.value}")
                )
            elif isinstance(constraint, LowerBoundInclusive):
                exprs.append(
                    (
                        pl.col(feature.name) >= constraint.value,
                        f"LowerBoundInclusive {feature.name} {constraint.value}",
                    )
                )
            elif isinstance(constraint, UpperBound):
                exprs.append(
                    (pl.col(feature.name) < constraint.value, f"UpperBound {feature.name} {constraint.value}")
                )
            elif isinstance(constraint, UpperBoundInclusive):
                exprs.append(
                    (
                        pl.col(feature.name) <= constraint.value,
                        f"UpperBoundInclusive {feature.name} {constraint.value}",
                    )
                )
            elif isinstance(constraint, InDomain):
                exprs.append(
                    (
                        pl.col(feature.name).is_in(constraint.values),
                        f"InDomain {feature.name} {constraint.values}",
                    )
                )
            elif isinstance(constraint, MinLength):
                exprs.append(
                    (
                        pl.col(feature.name).str.len_chars() > constraint.value,
                        f"MinLength {feature.name} {constraint.value}",
                    )
                )
            elif isinstance(constraint, MaxLength):
                exprs.append(
                    (
                        pl.col(feature.name).str.len_chars() < constraint.value,
                        f"MaxLength {feature.name} {constraint.value}",
                    )
                )
            elif isinstance(constraint, EndsWith):
                exprs.append(
                    (
                        pl.col(feature.name).str.ends_with(constraint.value),
                        f"EndsWith {feature.name} {constraint.value}",
                    )
                )
            elif isinstance(constraint, StartsWith):
                exprs.append(
                    (
                        pl.col(feature.name).str.starts_with(constraint.value),
                        f"StartsWith {feature.name} {constraint.value}",
                    )
                )
            elif isinstance(constraint, Regex):
                exprs.append(
                    (
                        pl.col(feature.name).str.contains(constraint.value),
                        f"Regex {feature.name} {constraint.value}",
                    )
                )

    return exprs


@dataclass
class ReturnInvalidJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    should_return_validation: bool

    def describe(self) -> str:
        expressions = [
            expr.not_().alias(f"not {name}")
            for expr, name in polars_filter_expressions_from(list(self.request_result.features))
        ]

        return 'ReturnInvalidJob ' + self.job.describe() + ' with filter expressions ' + str(expressions)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        raw_exprs = polars_filter_expressions_from(list(self.request_result.features))
        expressions = [expr.not_().alias(f"not {name}") for expr, name in raw_exprs]

        if self.should_return_validation:
            condition_cols = [f"not {name}" for _, name in raw_exprs]
            return (
                (await self.job.to_lazy_polars())
                .with_columns(expressions)
                .filter(pl.any_horizontal(*condition_cols))
            )
        else:
            return (await self.job.to_lazy_polars()).filter(pl.any_horizontal(expressions))

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()


CustomPolarsTransform = (
    Callable[[pl.LazyFrame], pl.LazyFrame] | Callable[[pl.LazyFrame], Coroutine[None, None, pl.LazyFrame]]
)  # noqa: E501


@dataclass
class CustomPolarsJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    polars_function: CustomPolarsTransform

    async def to_lazy_polars(self) -> pl.LazyFrame:
        import inspect

        df = await self.job.to_lazy_polars()

        if inspect.iscoroutinefunction(self.polars_function):
            return await self.polars_function(df)
        else:
            return self.polars_function(df)  # type: ignore

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
class SplitConfig:

    left_size: float
    right_size: float

    event_timestamp_column: str | None = None
    target_columns: list[str] | None = None


@dataclass
class InMemorySplitCacheJob(RetrivalJob, ModificationJob):

    job: RetrivalJob

    splitter: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]
    dataset_sizes: tuple[float, float]
    dataset_index: int

    cached_data: tuple[pl.DataFrame, pl.DataFrame] | None = None

    @property
    def fraction(self) -> float:
        return self.dataset_sizes[self.dataset_index]

    async def to_lazy_polars(self) -> pl.LazyFrame:
        cache = self

        if isinstance(self.job, InMemorySplitCacheJob):
            cache = self.job

        if cache.cached_data is not None:
            return cache.cached_data[self.dataset_index].lazy()

        data = (await self.job.to_lazy_polars()).collect()
        self.cached_data = self.splitter(data)
        return self.cached_data[self.dataset_index].lazy()

    def with_dataset_index(self, dataset_index: int) -> InMemorySplitCacheJob:
        return InMemorySplitCacheJob(self, self.splitter, self.dataset_sizes, dataset_index, self.cached_data)

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
class StackSourceColumn:
    top_source_name: str
    bottom_source_name: str
    source_column: str


@dataclass
class StackJob(RetrivalJob):

    top: RetrivalJob
    bottom: RetrivalJob

    source_column: StackSourceColumn | None

    @property
    def request_result(self) -> RequestResult:
        return self.top.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return RetrivalRequest.combine(self.top.retrival_requests + self.bottom.retrival_requests)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        top = await self.top.to_lazy_polars()
        bottom = await self.bottom.to_lazy_polars()

        if self.source_column:
            top = top.with_columns(
                pl.lit(self.source_column.top_source_name).alias(self.source_column.source_column)
            )
            bottom = bottom.with_columns(
                pl.lit(self.source_column.bottom_source_name).alias(self.source_column.source_column)
            )

        return top.select(top.columns).collect().vstack(bottom.select(top.columns).collect()).lazy()

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    def describe(self) -> str:
        return f'Stacking {self.top.describe()} on top of {self.bottom.describe()}'


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

    def log_each_job(self, logger_func: Callable[[object], None] | None = None) -> RetrivalJob:
        sub_log = JoinAsofJob(
            left_job=self.left_job.log_each_job(logger_func),
            right_job=self.right_job.log_each_job(logger_func),
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

    method: Literal['inner', 'left', 'outer']
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
            polars_types = [
                feature
                for feature in return_request.features.union(return_request.entities)
                if feature.name == left_col
            ]
            if not polars_types:
                raise ValueError(f'Unable to find {left_col} in left request {return_request}.')

            polars_type = polars_types[0].dtype.polars_type

            left_column_dtypes = dict(zip(left.columns, left.dtypes))
            right_column_dtypes = dict(zip(right.columns, right.dtypes))

            if not left_column_dtypes[left_col].is_(polars_type):
                left = left.with_columns(pl.col(left_col).cast(polars_type))

            if not right_column_dtypes[right_col].is_(polars_type):
                right = right.with_columns(pl.col(right_col).cast(polars_type))

        return left.join(right, left_on=self.left_on, right_on=self.right_on, how=self.method)

    def log_each_job(self, logger_func: Callable[[object], None] | None = None) -> RetrivalJob:
        sub_log = JoinJobs(
            method=self.method,
            left_job=self.left_job.log_each_job(logger_func),
            right_job=self.right_job.log_each_job(logger_func),
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
    condition: Feature | str | pl.Expr | DerivedFeature

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()

        if isinstance(self.condition, str):
            try:
                col = pl.Expr.deserialize(StringIO(self.condition))
            except Exception:
                col = pl.col(self.condition)
        elif isinstance(self.condition, pl.Expr):
            col = self.condition
        elif isinstance(self.condition, DerivedFeature):
            from aligned.feature_store import ContractStore

            store = ContractStore.empty()
            expr = await self.condition.transformation.transform_polars(df, self.condition.name, store)
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

        if isinstance(self.condition, pl.Expr):
            return (await self.to_polars()).to_pandas()
        if isinstance(self.condition, str):
            mask = df[self.condition]
        elif isinstance(self.condition, DerivedFeature):
            from aligned.feature_store import ContractStore

            store = ContractStore.empty()
            mask = await self.condition.transformation.transform_pandas(df, store)
        elif isinstance(self.condition, Feature):
            mask = df[self.condition.name]
        else:
            raise ValueError()

        return df.loc[mask]

    def describe(self) -> str:
        return f'{self.job.describe()} -> Filter based on {self.condition}'


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
    def loaded_columns(self) -> list[str]:
        return list(self.data.keys())

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
    logger: Callable[[object], None] = field(default=logger.debug)

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
        self.logger(f'Starting to run {type(self.job).__name__} - {job_name}')
        try:
            df = await self.job.to_pandas()
        except Exception as error:
            self.logger(f'Failed in job: {type(self.job).__name__} - {job_name}')
            raise error
        self.logger(f'Results from {type(self.job).__name__} - {job_name}')
        self.logger(df.columns)
        self.logger(df.head())
        return df

    async def to_lazy_polars(self) -> pl.LazyFrame:
        if logger.level == 0:
            logging.basicConfig(level=logging.DEBUG)

        job_name = self.retrival_requests[0].name
        self.logger(f'Starting to run {type(self.job).__name__} - {job_name}')
        try:
            df = await self.job.to_lazy_polars()
        except Exception as error:
            self.logger(f'Failed in job: {type(self.job).__name__} - {job_name}')
            raise error
        self.logger(f'Results from {type(self.job).__name__} - {job_name}')
        self.logger(df.columns)
        self.logger(df)
        self.logger(df.head(10).collect())
        return df

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()

    def log_each_job(self, logger_func: Callable[[object], None] | None = None) -> RetrivalJob:
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
    store: ContractStore | None = field(default=None)

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    def inject_store(self, store: ContractStore) -> RetrivalJob:
        job = self.copy_with(self.job.inject_store(store))
        job.store = store
        return job

    def filter(self, condition: str | Feature | DerivedFeature | pl.Expr) -> RetrivalJob:

        if isinstance(condition, str):
            column_name = condition
        elif isinstance(condition, pl.Expr):
            column_name = condition.meta.output_name(raise_if_undetermined=False)
        else:
            column_name = condition.name

        if column_name is None:
            return FilteredJob(self, condition)

        if any(
            column_name in [feature.name for feature in request.derived_features] for request in self.requests
        ):
            return FilteredJob(self, condition)

        return self.copy_with(self.job.filter(condition))

    async def compute_derived_features_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        from aligned.feature_store import ContractStore

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

                    method = await feature.transformation.transform_polars(
                        df, feature.name, self.store or ContractStore.empty()
                    )
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
        from aligned.feature_store import ContractStore

        for request in self.requests:
            for feature_round in request.derived_features_order():
                for feature in feature_round:
                    if feature.transformation.should_skip(feature.name, list(df.columns)):
                        logger.debug(f'Skipping to compute {feature.name} as it is aleady computed')
                        continue

                    logger.debug(f'Computing feature with pandas: {feature.name}')
                    df[feature.name] = await feature.transformation.transform_pandas(
                        df[feature.depending_on_names], self.store or ContractStore.empty()  # type: ignore
                    )
        return df

    async def to_pandas(self) -> pd.DataFrame:
        return await self.compute_derived_features_pandas(await self.job.to_pandas())

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return await self.compute_derived_features_polars(await self.job.to_lazy_polars())

    def drop_invalid(self, validator: Validator | None = None) -> RetrivalJob:
        # We may validate a derived feature, so the validation should not propegate lower.
        # Or it can, but then we need to do multi stage validation.
        # So this is a temporary solution.
        return DropInvalidJob(self, validator or PolarsValidator())

    def remove_derived_features(self) -> RetrivalJob:
        new_requests = []
        for req in self.job.retrival_requests:
            new_requests.append(
                RetrivalRequest(
                    req.name,
                    location=req.location,
                    features=req.features,
                    entities=req.entities,
                    derived_features=set(),
                    aggregated_features=req.aggregated_features,
                    event_timestamp_request=req.event_timestamp_request,
                )
            )
        return self.job.without_derived_features().with_request(new_requests)


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

            assert window.window is not None
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

            agg_data = window_data.lazy().group_by(window.group_by_names).agg(agg_expr).collect()
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
                LiteralRetrivalJob(df.lazy(), needed_requests)
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
    job: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    async def to_pandas(self) -> pd.DataFrame:
        from aligned.local.job import FileFullJob
        from aligned.sources.local import LiteralReference

        assert isinstance(self.job, DerivedFeatureJob)
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
class LoadedAtJob(RetrivalJob, ModificationJob):

    job: RetrivalJob
    request: RetrivalRequest

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.job.retrival_requests

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_pandas()
        if not self.request.event_timestamp:
            return df

        name = self.request.event_timestamp.name
        timezone_name = self.request.event_timestamp.dtype.datetime_timezone
        if timezone_name:
            tz = timezone(timezone_name)
            df[name] = datetime.now(tz=tz)
        else:
            df[name] = datetime.now()
        return df

    async def to_lazy_polars(self) -> pl.LazyFrame:

        df = await self.job.to_lazy_polars()
        if not self.request.event_timestamp:
            return df

        name = self.request.event_timestamp.name
        timezone_name = self.request.event_timestamp.dtype.datetime_timezone
        if timezone_name:
            tz = timezone(timezone_name)
            col = pl.lit(datetime.now(tz=tz))
        else:
            col = pl.lit(datetime.now())
        return df.with_columns(col.alias(name))


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
        df = await self.to_polars()
        return df.to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()
        org_schema = dict(df.schema)
        for request in self.requests:
            features_to_check = request.all_required_features

            if request.aggregated_features:
                features_to_check.update({feature.derived_feature for feature in request.aggregated_features})

            if request.event_timestamp:
                features_to_check.add(request.event_timestamp.as_feature())

            for feature in features_to_check:

                if feature.name not in org_schema:
                    continue

                if feature.dtype.polars_type.is_(org_schema[feature.name]):
                    logger.debug(f'Skipping feature {feature.name}, already correct type')
                    continue

                if feature.dtype == FeatureType.boolean():
                    df = df.with_columns(pl.col(feature.name).cast(pl.Int8).cast(pl.Boolean))
                elif feature.dtype.is_array:
                    dtype = df.select(feature.name).dtypes[0]
                    if dtype == pl.Utf8:
                        df = df.with_columns(pl.col(feature.name).str.json_decode(pl.List(pl.Utf8)))
                elif feature.dtype.is_embedding:
                    dtype = df.select(feature.name).dtypes[0]
                    if dtype == pl.Utf8:
                        df = df.with_columns(pl.col(feature.name).str.json_decode(pl.List(pl.Float64)))
                elif (feature.dtype == FeatureType.json()) or feature.dtype.is_datetime:
                    logger.debug(f'Converting {feature.name} to {feature.dtype.name}')
                    pass
                else:
                    logger.debug(
                        f'Converting {feature.name} to {feature.dtype.name} - {feature.dtype.polars_type}'
                    )
                    df = df.with_columns(pl.col(feature.name).cast(feature.dtype.polars_type, strict=False))

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
        id = Int32().as_entity()
        a = Int32()

    class OtherView(FeatureView):
        metadata = FeatureViewMetadata(
            name="other_view",
            batch_source=FileSource.parquet_at("other.parquet")
        )
        id = Int32().as_entity()
        c = Int32()

    class Combined(CombinedFeatureView):
        metadata = CombinedMetadata(name="combined")

        some = SomeView()
        other = OtherView()

        added = some.a + other.c
    """

    jobs: list[RetrivalJob]
    combined_requests: list[RetrivalRequest]
    store: ContractStore | None = field(default=None)

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

    def inject_store(self, store: ContractStore) -> RetrivalJob:
        return CombineFactualJob(
            [job.inject_store(store) for job in self.jobs], self.combined_requests, store=store
        )

    async def combine_data(self, df: pd.DataFrame) -> pd.DataFrame:
        from aligned import ContractStore

        for request in self.combined_requests:
            for feature in request.derived_features:
                if feature.name in df.columns:
                    logger.debug(f'Skipping feature {feature.name}, already computed')
                    continue
                logger.debug(f'Computing feature: {feature.name}')
                df[feature.name] = await feature.transformation.transform_pandas(
                    df[feature.depending_on_names], self.store or ContractStore.empty()  # type: ignore
                )
        return df

    async def combine_polars_data(self, df: pl.LazyFrame) -> pl.LazyFrame:
        from aligned import ContractStore

        for request in self.combined_requests:
            logger.debug(f'{request.name}, {len(request.derived_features)}')
            for feature in request.derived_features:
                if feature.name in df.columns:
                    logger.debug(f'Skipping feature {feature.name}, already computed')
                    continue
                logger.debug(f'Computing feature: {feature.name}')
                result = await feature.transformation.transform_polars(
                    df, feature.name, self.store or ContractStore.empty()
                )
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

    def log_each_job(self, logger_func: Callable[[object], None] | None = None) -> RetrivalJob:
        return LogJob(
            CombineFactualJob([job.log_each_job(logger_func) for job in self.jobs], self.combined_requests)
        )

    def describe(self) -> str:
        description = f'Combining {len(self.jobs)} jobs:\n'
        for index, job in enumerate(self.jobs):
            description += f'{index + 1}: {job.describe()}\n'
        return description


@dataclass
class SelectColumnsJob(RetrivalJob, ModificationJob):

    include_features: list[str]
    job: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return self.job.request_result.filter_features(set(self.include_features))

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return [request.filter_features(set(self.include_features)) for request in self.job.retrival_requests]

    async def to_pandas(self) -> pd.DataFrame:
        df = await self.job.to_pandas()
        if self.include_features:
            total_list = list({ent.name for ent in self.request_result.entities}.union(self.include_features))
            return df[total_list]  # type: ignore
        else:
            return df

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()
        if self.include_features:
            total_list = list({ent.name for ent in self.request_result.entities}.union(self.include_features))
            return df.select(total_list)
        else:
            return df

    def cached_at(self, location: DataFileReference | str) -> RetrivalJob:
        return SelectColumnsJob(self.include_features, self.job.cached_at(location))

    def with_subfeatures(self) -> RetrivalJob:
        return self.job.with_subfeatures()

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()

    def ignore_event_timestamp(self) -> RetrivalJob:
        return SelectColumnsJob(
            include_features=list(set(self.include_features) - {'event_timestamp'}),
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

    def describe(self) -> str:
        return f"Custom Lazy Polars Job returning {self.request.all_returned_columns}"

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request(self.request)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return await self.method()

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_polars()).to_pandas()


@dataclass
class UnpackEmbedding(RetrivalJob, ModificationJob):

    job: RetrivalJob

    async def to_lazy_polars(self) -> pl.LazyFrame:
        df = await self.job.to_lazy_polars()

        list_features = [
            feature.name for feature in self.request_result.features if feature.dtype.is_embedding
        ]

        df = df.with_columns(
            [pl.col(feature).list.to_struct(n_field_strategy='max_width') for feature in list_features]
        )
        df = df.unnest(list_features)

        return df.select(pl.all().exclude(list_features))

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_polars()).to_pandas()


@dataclass
class PredictionJob(RetrivalJob):

    job: RetrivalJob
    model: Model
    store: ContractStore
    output_requests: list[RetrivalRequest]

    def added_features(self) -> set[Feature]:
        pred_view = self.model.predictions_view
        added = pred_view.features
        return added

    @property
    def request_result(self) -> RequestResult:
        reqs = self.retrival_requests
        return RequestResult.from_request_list(reqs)

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.output_requests + [self.model.predictions_view.request(self.model.name)]

    def describe(self) -> str:
        added = self.added_features()
        feature_names = {feat.name for feat in added}
        return (
            f"{self.job.describe()} \n"
            f"-> predicting using model {self.model.name} with {feature_names} added features"
        )

    async def to_pandas(self) -> pd.DataFrame:
        return await self.job.to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        from aligned.exposed_model.interface import VersionedModel
        from datetime import datetime, timezone

        predictor = self.model.exposed_model
        if not predictor:
            raise ValueError('No predictor defined for model')

        output = self.model.predictions_view
        model_version_column = output.model_version_column

        df = await predictor.run_polars(
            self.job,
            self.store.model(self.model.name),
        )
        if output.event_timestamp and output.event_timestamp.name not in df.columns:
            df = df.with_columns(
                pl.lit(datetime.now(timezone.utc)).alias(output.event_timestamp.name),
            )

        if (
            model_version_column
            and isinstance(predictor, VersionedModel)
            and model_version_column.name not in df.columns
        ):
            df = df.with_columns(
                pl.lit(await predictor.model_version()).alias(model_version_column.name),
            )
        return df.lazy()

    def log_each_job(self, logger_func: Callable[[object], None] | None = None) -> RetrivalJob:
        return PredictionJob(self.job.log_each_job(logger_func), self.model, self.store, self.output_requests)

    def filter(self, condition: str | Feature | DerivedFeature | pl.Expr) -> RetrivalJob:
        return PredictionJob(self.job.filter(condition), self.model, self.store, self.output_requests)

    def remove_derived_features(self) -> RetrivalJob:
        return self.job.remove_derived_features()

    async def insert_into_output_source(self) -> None:
        from aligned.feature_source import WritableFeatureSource

        pred_source = self.model.predictions_view.source
        if not pred_source:
            raise ValueError('No source defined for predictions view')

        if not isinstance(pred_source, WritableFeatureSource):
            raise ValueError('Source for predictions view is not writable')

        req = self.model.predictions_view.request('preds')
        await pred_source.insert(self, req)

    async def upsert_into_output_source(self) -> None:
        from aligned.feature_source import WritableFeatureSource

        pred_source = self.model.predictions_view.source
        if not pred_source:
            raise ValueError('No source defined for predictions view')

        if not isinstance(pred_source, WritableFeatureSource):
            raise ValueError('Source for predictions view is not writable')

        req = self.model.predictions_view.request('preds')
        await pred_source.upsert(self, req)

    async def overwrite_output_source(self) -> None:
        from aligned.feature_source import WritableFeatureSource

        pred_source = self.model.predictions_view.source
        if not pred_source:
            raise ValueError('No source defined for predictions view')

        if not isinstance(pred_source, WritableFeatureSource):
            raise ValueError('Source for predictions view is not writable')

        req = self.model.predictions_view.request('preds')
        await pred_source.overwrite(self, req)
