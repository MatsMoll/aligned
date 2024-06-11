from __future__ import annotations
from copy import copy
from datetime import timedelta, timezone, datetime

from typing import TYPE_CHECKING, Awaitable, TypeVar, Any, Callable, Coroutine
from dataclasses import dataclass

from mashumaro.types import SerializableType
from aligned.data_file import DataFileReference

from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import EventTimestamp, Feature, FeatureLocation, FeatureType
from aligned.request.retrival_request import RequestResult, RetrivalRequest
from aligned.compiler.feature_factory import FeatureFactory
from polars.type_aliases import TimeUnit
import polars as pl

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aligned.retrival_job import RetrivalJob

T = TypeVar('T')


class BatchDataSourceFactory:

    supported_data_sources: dict[str, type[BatchDataSource]]

    _shared: BatchDataSourceFactory | None = None

    def __init__(self) -> None:
        from aligned.sources.local import (
            CsvFileSource,
            ParquetFileSource,
            DeltaFileSource,
            PartitionedParquetFileSource,
        )
        from aligned.schemas.feature_view import FeatureViewReferenceSource
        from aligned.schemas.model import ModelSource

        from aligned.sources.psql import PostgreSQLDataSource
        from aligned.sources.redshift import RedshiftSQLDataSource
        from aligned.sources.s3 import AwsS3CsvDataSource, AwsS3ParquetDataSource
        from aligned.sources.azure_blob_storage import (
            AzureBlobCsvDataSource,
            AzureBlobDeltaDataSource,
            AzureBlobParquetDataSource,
            AzureBlobPartitionedParquetDataSource,
        )
        from aligned.sources.lancedb import LanceDbTable

        source_types = [
            PostgreSQLDataSource,
            # File Sources
            CsvFileSource,
            DeltaFileSource,
            ParquetFileSource,
            PartitionedParquetFileSource,
            # Aws Sources
            AwsS3CsvDataSource,
            AwsS3ParquetDataSource,
            RedshiftSQLDataSource,
            # Azure Sources
            AzureBlobCsvDataSource,
            AzureBlobDeltaDataSource,
            AzureBlobParquetDataSource,
            AzureBlobPartitionedParquetDataSource,
            # LanceDB
            LanceDbTable,
            # Aligned Related Sources
            JoinDataSource,
            JoinAsofDataSource,
            FilteredDataSource,
            FeatureViewReferenceSource,
            CustomMethodDataSource,
            ModelSource,
            StackSource,
        ]

        self.supported_data_sources = {source.type_name: source for source in source_types}

    @classmethod
    def shared(cls) -> BatchDataSourceFactory:
        if cls._shared:
            return cls._shared
        cls._shared = BatchDataSourceFactory()
        return cls._shared


class BatchSourceModification:

    source: BatchDataSource

    def wrap_job(self, job: RetrivalJob) -> RetrivalJob:
        raise NotImplementedError()


class BatchDataSource(Codable, SerializableType):
    """
    A definition to where a specific pice of data can be found.
    E.g: A database table, a file, a web service, etc.

    Ths can thereafter be combined with other BatchDataSources in order to create a rich dataset.
    """

    type_name: str

    def job_group_key(self) -> str:
        """
        A key defining which sources can be grouped together in one request.
        """
        raise NotImplementedError(type(self))

    def source_id(self) -> str:
        """
        An id that identifies a source from others.
        """
        return self.job_group_key()

    def _serialize(self) -> dict:
        assert (
            self.type_name in BatchDataSourceFactory.shared().supported_data_sources
        ), f'Unknown type_name: {self.type_name}'
        return self.to_dict()

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    def transform_with_polars(
        self,
        method: Callable[[pl.LazyFrame], Awaitable[pl.LazyFrame]] | Callable[[pl.LazyFrame], pl.LazyFrame],
    ) -> BatchDataSource:
        async def all(request: RetrivalRequest, limit: int | None) -> pl.LazyFrame:
            import inspect

            df = await self.all_data(request, limit).to_lazy_polars()

            if inspect.iscoroutinefunction(method):
                return await method(df)
            else:
                return method(df)

        async def all_between_dates(
            request: RetrivalRequest, start_date: datetime, end_date: datetime
        ) -> pl.LazyFrame:
            import inspect

            df = await self.all_between_dates(request, start_date, end_date).to_lazy_polars()

            if inspect.iscoroutinefunction(method):
                return await method(df)
            else:
                return method(df)

        async def features_for(entities: RetrivalJob, request: RetrivalRequest) -> pl.LazyFrame:
            import inspect

            df = await self.features_for(entities, request).to_lazy_polars()

            if inspect.iscoroutinefunction(method):
                return await method(df)
            else:
                return method(df)

        return CustomMethodDataSource.from_methods(
            all_data=all,
            all_between_dates=all_between_dates,
            features_for=features_for,
            depends_on_sources=self.location_id(),
        )

    def contains_config(self, config: Any) -> bool:
        """
        Checks if a data source contains a source config.
        This can be used to select different sources based on the data sources to connect to.

        ```
        config = PostgreSQLConfig(env_var='MY_APP_DB_URL')
        source = config.table('my_table')

        print(source.contains_config(config))
        >> True

        store = await FileSource.json_at("features.json").feature_store()
        views = store.views_with_config(config)
        print(len(views))
        >> 3
        ```

        Args:
            config: The config to check for

        Returns:
            bool: If the config is contained in the source
        """
        if isinstance(config, BatchDataSource):
            return config.to_dict() == self.to_dict()
        return False

    @classmethod
    def _deserialize(cls, value: dict) -> BatchDataSource:
        name_type = value['type_name']
        if name_type not in BatchDataSourceFactory.shared().supported_data_sources:
            raise ValueError(
                f"Unknown batch data source id: '{name_type}'.\nRemember to add the"
                ' data source to the BatchDataSourceFactory.supported_data_sources if'
                ' it is a custom type.'
            )
        del value['type_name']
        data_class = BatchDataSourceFactory.shared().supported_data_sources[name_type]
        return data_class.from_dict(value)

    def all_columns(self, limit: int | None = None) -> RetrivalJob:
        return self.all(RequestResult(set(), set(), None), limit=limit)

    def all(self, result: RequestResult, limit: int | None = None) -> RetrivalJob:
        return self.all_data(
            result.as_retrival_request('read_all', location=FeatureLocation.feature_view('read_all')),
            limit=limit,
        )

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:
        if isinstance(self, BatchSourceModification):
            return self.wrap_job(self.source.all_data(request, limit))

        if isinstance(self, DataFileReference):
            from aligned.local.job import FileFullJob

            return FileFullJob(self, request=request, limit=limit)

        raise NotImplementedError(type(self))

    def all_between_dates(
        self,
        request: RetrivalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrivalJob:

        if isinstance(self, BatchSourceModification):
            return self.wrap_job(self.source.all_between_dates(request, start_date, end_date))

        if isinstance(self, DataFileReference):
            from aligned.local.job import FileDateJob

            return FileDateJob(self, request=request, start_date=start_date, end_date=end_date)

        raise NotImplementedError(type(self))

    @classmethod
    def multi_source_features_for(
        cls: type[T], facts: RetrivalJob, requests: list[tuple[T, RetrivalRequest]]
    ) -> RetrivalJob:

        sources = {source.job_group_key() for source, _ in requests if isinstance(source, BatchDataSource)}
        if len(sources) != 1:
            raise NotImplementedError(
                f'Type: {cls} have not implemented how to load fact data with multiple sources.'
            )

        source, _ = requests[0]
        if isinstance(source, BatchSourceModification):
            return source.wrap_job(
                type(source.source).multi_source_features_for(facts, requests)  # type: ignore
            )
        elif isinstance(source, DataFileReference):
            from aligned.local.job import FileFactualJob

            return FileFactualJob(source, [request for _, request in requests], facts)
        else:
            raise NotImplementedError(f'Type: {cls} have not implemented how to load fact data')

    def features_for(self, facts: RetrivalJob, request: RetrivalRequest) -> RetrivalJob:
        return type(self).multi_source_features_for(facts, [(self, request)])

    async def schema(self) -> dict[str, FeatureType]:
        """Returns the schema for the data source

        ```python
        source = FileSource.parquet_at('test_data/titanic.parquet')
        schema = await source.schema()
        >>> {'passenger_id': FeatureType(name='int64'), ...}
        ```

        Returns:
            dict[str, FeatureType]: A dictionary containing the column name and the feature type
        """
        if isinstance(self, BatchSourceModification):
            return await self.source.schema()
        raise NotImplementedError(f'`schema()` is not implemented for {type(self)}.')

    async def feature_view_code(self, view_name: str) -> str:
        """Setup the code needed to represent the data source as a feature view

        ```python
        FileSource.parquet("my_path.parquet").feature_view_code(view_name="my_view")

        >>> \"\"\"from aligned import FeatureView, String, Int64, Float

        class MyView(FeatureView):

            metadata = FeatureView.metadata_with(
                name="Embarked",
                description="some description",
                batch_source=FileSource.parquest("my_path.parquet")
                stream_source=None,
            )

            Passenger_id = Int64()
            Survived = Int64()
            Pclass = Int64()
            Name = String()
            Sex = String()
            Age = Float()
            Sibsp = Int64()
            Parch = Int64()
            Ticket = String()
            Fare = Float()
            Cabin = String()
            Embarked = String()\"\"\"
        ```

        Returns:
            str: The code needed to setup a basic feature view
        """
        from aligned.feature_view.feature_view import FeatureView

        schema = await self.schema()
        feature_types = {name: feature_type.feature_factory for name, feature_type in schema.items()}
        return FeatureView.feature_view_code_template(feature_types, f'{self}', view_name)

    async def freshness(self, event_timestamp: EventTimestamp) -> datetime | None:
        """
        my_table_freshenss = await (PostgreSQLConfig("DB_URL")
            .table("my_table")
            .freshness()
        )
        """
        from aligned.data_file import DataFileReference
        from aligned.sources.local import data_file_freshness

        if isinstance(self, DataFileReference):
            return await data_file_freshness(self, event_timestamp.name)

        raise NotImplementedError(f'Freshness is not implemented for {type(self)}.')

    def filter(self, condition: DerivedFeature | Feature) -> BatchDataSource:
        return FilteredDataSource(self, condition)

    def location_id(self) -> set[FeatureLocation]:
        return self.depends_on()

    def depends_on(self) -> set[FeatureLocation]:
        return set()

    def tags(self) -> list[str]:
        return [self.type_name]


@dataclass
class CustomMethodDataSource(BatchDataSource):

    all_data_method: bytes
    all_between_dates_method: bytes
    features_for_method: bytes
    depends_on_sources: set[FeatureLocation] | None = None

    type_name: str = 'custom_method'

    def job_group_key(self) -> str:
        return 'custom_method'

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:
        from aligned.retrival_job import CustomLazyPolarsJob
        import dill

        return CustomLazyPolarsJob(
            request=request, method=lambda: dill.loads(self.all_data_method)(request, limit)
        ).fill_missing_columns()

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:
        from aligned.retrival_job import CustomLazyPolarsJob
        import dill

        return CustomLazyPolarsJob(
            request=request,
            method=lambda: dill.loads(self.all_between_dates_method)(request, start_date, end_date),
        ).fill_missing_columns()

    def features_for(self, facts: RetrivalJob, request: RetrivalRequest) -> RetrivalJob:
        from aligned.retrival_job import CustomLazyPolarsJob
        import dill

        return CustomLazyPolarsJob(
            request=request, method=lambda: dill.loads(self.features_for_method)(facts, request)
        ).fill_missing_columns()

    @classmethod
    def multi_source_features_for(
        cls: type[T], facts: RetrivalJob, requests: list[tuple[T, RetrivalRequest]]
    ) -> RetrivalJob:

        if len(requests) != 1:
            raise NotImplementedError(
                f'Type: {cls} have not implemented how to load fact data with multiple sources.'
            )

        source, request = requests[0]
        return source.features_for(facts, request)  # type: ignore

    @staticmethod
    def from_methods(
        all_data: Callable[[RetrivalRequest, int | None], Coroutine[None, None, pl.LazyFrame]] | None = None,
        all_between_dates: Callable[
            [RetrivalRequest, datetime, datetime], Coroutine[None, None, pl.LazyFrame]
        ]
        | None = None,
        features_for: Callable[[RetrivalJob, RetrivalRequest], Coroutine[None, None, pl.LazyFrame]]
        | None = None,
        depends_on_sources: set[FeatureLocation] | None = None,
    ) -> 'CustomMethodDataSource':
        import dill

        if not all_data:
            all_data = CustomMethodDataSource.default_throw  # type: ignore

        if not all_between_dates:
            all_between_dates = CustomMethodDataSource.default_throw  # type: ignore

        if not features_for:
            features_for = CustomMethodDataSource.default_throw  # type: ignore

        return CustomMethodDataSource(
            all_data_method=dill.dumps(all_data),
            all_between_dates_method=dill.dumps(all_between_dates),
            features_for_method=dill.dumps(features_for),
            depends_on_sources=depends_on_sources,
        )

    @staticmethod
    def default_throw(**kwargs: Any) -> pl.LazyFrame:
        raise NotImplementedError('No method is defined for this data source.')

    def depends_on(self) -> set[FeatureLocation]:
        return self.depends_on_sources or set()


@dataclass
class FilteredDataSource(BatchDataSource):

    source: BatchDataSource
    condition: DerivedFeature | Feature

    type_name: str = 'subset'

    def job_group_key(self) -> str:
        return f'subset/{self.source.job_group_key()}'

    async def schema(self) -> dict[str, FeatureType]:
        return await self.source.schema()

    @classmethod
    def multi_source_features_for(
        cls: type[FilteredDataSource],
        facts: RetrivalJob,
        requests: list[tuple[FilteredDataSource, RetrivalRequest]],
    ) -> RetrivalJob:

        sources = {source.job_group_key() for source, _ in requests if isinstance(source, BatchDataSource)}
        if len(sources) != 1:
            raise NotImplementedError(
                f'Type: {cls} have not implemented how to load fact data with multiple sources.'
            )
        source, request = requests[0]

        if isinstance(source.condition, DerivedFeature):
            request.derived_features.add(source.condition)
        else:
            request.features.add(source.condition)

        return source.source.features_for(facts, request).filter(source.condition)

    async def freshness(self, event_timestamp: EventTimestamp) -> datetime | None:
        return await self.source.freshness(event_timestamp)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:

        if isinstance(self.condition, Feature):
            request.features.add(self.condition)
        else:
            request.derived_features.add(self.condition)

        return (
            self.source.all_between_dates(request, start_date, end_date)
            .filter(self.condition)
            .aggregate(request)
            .derive_features([request])
        )

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:

        if isinstance(self.condition, DerivedFeature):
            request.derived_features.add(self.condition)
        else:
            request.features.add(self.condition)

        return (
            self.source.all_data(request, limit)
            .filter(self.condition)
            .aggregate(request)
            .derive_features([request])
        )

    def depends_on(self) -> set[FeatureLocation]:
        return self.source.depends_on()


def resolve_keys(keys: str | FeatureFactory | list[str] | list[FeatureFactory]) -> list[str]:

    if isinstance(keys, FeatureFactory):
        return [keys.name]

    if isinstance(keys, str):
        return [keys]

    if isinstance(keys[0], FeatureFactory):
        return [key.name for key in keys]  # type: ignore

    return keys  # type: ignore


def model_prediction_instance_source(model: Any) -> tuple[BatchDataSource, RetrivalRequest] | Exception:
    from aligned.schemas.feature_view import FeatureViewReferenceSource
    from aligned.compiler.model import ModelContractWrapper

    if not hasattr(model, '__model_wrapper__'):
        return ValueError(
            f'Unable to join {model} as a __view_wrapper__ is needed. Make sure you have used @feature_view'
        )

    wrapper = getattr(model, '__model_wrapper__')
    if not isinstance(wrapper, ModelContractWrapper):
        return ValueError()

    compiled_view = wrapper.as_view()
    if compiled_view is None:
        return ValueError()

    return (
        FeatureViewReferenceSource(compiled_view, FeatureLocation.model(compiled_view.name)),
        compiled_view.request_all.needed_requests[0],
    )


def view_wrapper_instance_source(view: Any) -> tuple[BatchDataSource, RetrivalRequest] | Exception:
    from aligned.feature_view.feature_view import FeatureViewWrapper
    from aligned.schemas.feature_view import FeatureViewReferenceSource

    if not hasattr(view, '__view_wrapper__'):
        return ValueError(
            f'Unable to join {view} as a __view_wrapper__ is needed. Make sure you have used @feature_view'
        )

    wrapper = getattr(view, '__view_wrapper__')
    if not isinstance(wrapper, FeatureViewWrapper):
        return ValueError()

    compiled_view = wrapper.compile()

    return (
        FeatureViewReferenceSource(compiled_view, FeatureLocation.feature_view(compiled_view.name)),
        compiled_view.request_all.needed_requests[0],
    )


def join_asof_source(
    source: BatchDataSource,
    left_request: RetrivalRequest,
    view: Any,
    left_on: list[str] | None = None,
    right_on: list[str] | None = None,
) -> JoinAsofDataSource:

    wrapped_source = view_wrapper_instance_source(view)
    if isinstance(wrapped_source, Exception):
        wrapped_source = model_prediction_instance_source(view)

    if isinstance(wrapped_source, Exception):
        raise wrapped_source

    right_source, right_request = wrapped_source

    left_event_timestamp = left_request.event_timestamp
    right_event_timestamp = right_request.event_timestamp

    if left_event_timestamp is None:
        raise ValueError('A left event timestamp is needed, but found none.')
    if right_event_timestamp is None:
        raise ValueError('A right event timestamp is needed, but found none.')

    return JoinAsofDataSource(
        source=source,
        left_request=left_request,
        right_source=right_source,
        right_request=right_request,
        left_event_timestamp=left_event_timestamp.name,
        right_event_timestamp=right_event_timestamp.name,
        left_on=left_on,
        right_on=right_on,
    )


def join_source(
    source: BatchDataSource,
    view: Any,
    on_left: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
    on_right: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
    how: str = 'inner',
    left_request: RetrivalRequest | None = None,
) -> JoinDataSource:
    from aligned.data_source.batch_data_source import JoinDataSource
    from aligned.feature_view.feature_view import FeatureViewWrapper

    wrapped_source = view_wrapper_instance_source(view)
    if isinstance(wrapped_source, Exception):
        wrapped_source = model_prediction_instance_source(view)

    if isinstance(wrapped_source, Exception):
        raise wrapped_source

    right_source, right_request = wrapped_source

    if on_left is None:
        left_keys = list(right_request.entity_names)
    else:
        left_keys = resolve_keys(on_left)

    if on_right is None:
        right_keys = list(right_request.entity_names)
    else:
        right_keys = resolve_keys(on_right)

    if left_request is None:
        if isinstance(source, JoinDataSource):
            left_request = RetrivalRequest.unsafe_combine([source.left_request, source.right_request])
        elif isinstance(source, FeatureViewWrapper):
            left_request = source.compile().request_all.needed_requests[0]

    if left_request is None:
        raise ValueError('Unable to resolve the left request. Concider adding a `left_request` param.')

    return JoinDataSource(
        source=source,
        left_request=left_request,
        right_source=right_source,
        right_request=right_request,
        left_on=left_keys,
        right_on=right_keys,
        method=how,
    )


@dataclass
class JoinAsofDataSource(BatchDataSource):

    source: BatchDataSource
    left_request: RetrivalRequest
    right_source: BatchDataSource
    right_request: RetrivalRequest

    left_event_timestamp: str
    right_event_timestamp: str

    left_on: list[str] | None = None
    right_on: list[str] | None = None

    timestamp_unit: TimeUnit = 'us'

    type_name: str = 'join_asof'

    async def schema(self) -> dict[str, FeatureType]:
        left_schema = await self.source.schema()
        right_schema = await self.right_source.schema()

        return {**left_schema, **right_schema}

    def job_group_key(self) -> str:
        return f'join/{self.source.job_group_key()}'

    def all_with_limit(self, limit: int | None) -> RetrivalJob:

        right_job = self.right_source.all_data(self.right_request, limit=None).derive_features(
            [self.right_request]
        )

        return (
            self.source.all_data(self.left_request, limit=limit)
            .derive_features([self.left_request])
            .join_asof(
                right_job,
                left_event_timestamp=self.left_event_timestamp,
                right_event_timestamp=self.right_event_timestamp,
                left_on=self.left_on,
                right_on=self.right_on,
                timestamp_unit=self.timestamp_unit,
            )
            .fill_missing_columns()
        )

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:

        right_job = self.right_source.all_data(self.right_request, limit=None).derive_features(
            [self.right_request]
        )

        return (
            self.source.all_data(self.left_request, limit=limit)
            .derive_features([self.left_request])
            .join_asof(
                right_job,
                left_event_timestamp=self.left_event_timestamp,
                right_event_timestamp=self.right_event_timestamp,
                left_on=self.left_on,
                right_on=self.right_on,
                timestamp_unit=self.timestamp_unit,
            )
            .aggregate(request)
            .fill_missing_columns()
            .derive_features([request])
        )

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:

        right_job = self.right_source.all_data(self.right_request, limit=None).derive_features(
            [self.right_request]
        )

        return (
            self.source.all_between_dates(self.left_request, start_date, end_date)
            .derive_features([self.left_request])
            .join_asof(
                right_job,
                left_event_timestamp=self.left_event_timestamp,
                right_event_timestamp=self.right_event_timestamp,
                left_on=self.left_on,
                right_on=self.right_on,
            )
            .aggregate(request)
            .fill_missing_columns()
            .derive_features([request])
        )

    async def freshness(self, event_timestamp: EventTimestamp) -> datetime | None:
        left_freshness = await self.source.freshness(event_timestamp)
        right_frehsness = await self.right_source.freshness(event_timestamp)

        if left_freshness is None:
            return None

        if right_frehsness is None:
            return None

        return min(left_freshness, right_frehsness)

    def join(
        self,
        view: Any,
        on: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        how: str = 'inner',
    ) -> JoinDataSource:
        return join_source(self, view, on, how)

    def join_asof(
        self, view: Any, on: str | FeatureFactory | list[str] | list[FeatureFactory]
    ) -> JoinAsofDataSource:

        left_on = None
        right_on = None
        if on:
            left_on = resolve_keys(on)
            right_on = left_on

        left_request = RetrivalRequest.unsafe_combine([self.left_request, self.right_request])

        return join_asof_source(
            self, left_request=left_request, view=view, left_on=left_on, right_on=right_on
        )

    def depends_on(self) -> set[FeatureLocation]:
        return self.source.depends_on().union(self.right_source.depends_on())


@dataclass
class StackSource(BatchDataSource):

    top: BatchDataSource
    bottom: BatchDataSource

    source_column: str | None = None

    type_name: str = 'stack'

    @property
    def source_column_config(self):  # type: ignore
        from aligned.retrival_job import StackSourceColumn

        if not self.source_column:
            return None

        return StackSourceColumn(
            top_source_name=self.top.source_id(),
            bottom_source_name=self.bottom.source_id(),
            source_column=self.source_column,
        )

    def sub_request(self, request: RetrivalRequest, config) -> RetrivalRequest:  # type: ignore
        return RetrivalRequest(
            name=request.name,
            location=request.location,
            features={feature for feature in request.features if feature.name != config.source_column},
            entities=request.entities,
            derived_features={
                feature
                for feature in request.derived_features
                if not any(dep.name == config.source_column for dep in feature.depending_on)
            },
            aggregated_features=request.aggregated_features,
            event_timestamp_request=request.event_timestamp_request,
            features_to_include=request.features_to_include - {config.source_column},
        )

    def job_group_key(self) -> str:
        return f'stack/{self.top.job_group_key()}/{self.bottom.job_group_key()}'

    async def schema(self) -> dict[str, FeatureType]:
        top_schema = await self.top.schema()
        bottom_schema = await self.bottom.schema()

        return {**top_schema, **bottom_schema}

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:
        from aligned.retrival_job import StackJob

        config = self.source_column_config

        sub_request = request

        if config:
            sub_request = self.sub_request(request, config)

        return (
            StackJob(
                top=self.top.all_data(sub_request, int(limit / 2) if limit else None),
                bottom=self.bottom.all_data(sub_request, int(limit / 2) if limit else None),
                source_column=self.source_column_config,
            )
            .with_request([request])
            .derive_features([request])
        )

    @classmethod
    def multi_source_features_for(
        cls, facts: RetrivalJob, requests: list[tuple[StackSource, RetrivalRequest]]
    ) -> RetrivalJob:
        sources = {source.job_group_key() for source, _ in requests}
        if len(sources) != 1:
            raise ValueError(f'Only able to load one {requests} at a time')

        source = requests[0][0]
        if not isinstance(source, cls):
            raise ValueError(f'Only {cls} is supported, recived: {source}')

        return source.features_for(facts, requests[0][1])

    def features_for(self, facts: RetrivalJob, request: RetrivalRequest) -> RetrivalJob:
        from aligned.local.job import FileFactualJob
        from aligned.retrival_job import StackJob

        config = self.source_column_config
        sub_request = request

        if config:
            sub_request = self.sub_request(request, config)

        top = self.top.features_for(facts, sub_request)
        bottom = self.bottom.features_for(facts, sub_request)

        stack_job = StackJob(top=top, bottom=bottom, source_column=config)

        return FileFactualJob(stack_job, [request], facts)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:
        from aligned.retrival_job import StackJob

        top = self.top.all_between_dates(request, start_date, end_date)
        bottom = self.bottom.all_between_dates(request, start_date, end_date)

        return StackJob(
            top=top,
            bottom=bottom,
            source_column=self.source_column_config,
        )

    def depends_on(self) -> set[FeatureLocation]:
        return self.top.depends_on().union(self.bottom.depends_on())


@dataclass
class JoinDataSource(BatchDataSource):

    source: BatchDataSource
    left_request: RetrivalRequest
    right_source: BatchDataSource
    right_request: RetrivalRequest
    left_on: list[str]
    right_on: list[str]
    method: str

    type_name: str = 'join'

    async def schema(self) -> dict[str, FeatureType]:
        left_schema = await self.source.schema()
        right_schema = await self.right_source.schema()

        return {**left_schema, **right_schema}

    def job_group_key(self) -> str:
        return f'join/{self.source.job_group_key()}'

    def all_with_limit(self, limit: int | None) -> RetrivalJob:
        right_job = self.right_source.all_data(self.right_request, limit=None).derive_features(
            [self.right_request]
        )

        return (
            self.source.all_data(self.left_request, limit=limit)
            .derive_features([self.left_request])
            .join(right_job, method=self.method, left_on=self.left_on, right_on=self.right_on)
            .fill_missing_columns()
        )

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:

        right_job = self.right_source.all_data(self.right_request, limit=None).derive_features(
            [self.right_request]
        )

        return (
            self.source.all_data(self.left_request, limit=limit)
            .derive_features([self.left_request])
            .join(right_job, method=self.method, left_on=self.left_on, right_on=self.right_on)
            .fill_missing_columns()
            .aggregate(request)
            .derive_features([request])
        )

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:

        right_job = self.right_source.all_data(self.right_request, limit=None).derive_features(
            [self.right_request]
        )

        return (
            self.source.all_between_dates(self.left_request, start_date, end_date)
            .derive_features([self.left_request])
            .join_asof(
                right_job,
                left_event_timestamp=self.left_event_timestamp,
                right_event_timestamp=self.right_event_timestamp,
                left_on=self.left_on,
                right_on=self.right_on,
            )
            .fill_missing_columns()
            .aggregate(request)
            .derive_features([request])
        )

    async def freshness(self, event_timestamp: EventTimestamp) -> datetime | None:
        left_freshness = await self.source.freshness(event_timestamp)
        right_frehsness = await self.right_source.freshness(event_timestamp)

        if left_freshness is None:
            return None

        if right_frehsness is None:
            return None

        return min(left_freshness, right_frehsness)

    def join(
        self,
        view: Any,
        on: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        on_left: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        on_right: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        how: str = 'inner',
    ) -> JoinDataSource:

        if on:
            on_left = on
            on_right = on

        return join_source(self, view, on_left, on_right, how)

    def depends_on(self) -> set[FeatureLocation]:
        return self.source.depends_on().union(self.right_source.depends_on())


class ColumnFeatureMappable:
    mapping_keys: dict[str, str]

    def with_renames(self: T, mapping_keys: dict[str, str]) -> T:
        new = copy(self)
        new.mapping_keys = mapping_keys  # type: ignore
        return new

    def columns_for(self, features: list[Feature]) -> list[str]:
        return [self.mapping_keys.get(feature.name, feature.name) for feature in features]

    def feature_identifier_for(self, columns: list[str]) -> list[str]:
        reverse_map = {v: k for k, v in self.mapping_keys.items()}
        return [reverse_map.get(column, column) for column in columns]


def data_for_request(request: RetrivalRequest, size: int) -> pl.DataFrame:
    from aligned.schemas.constraints import (
        InDomain,
        LowerBound,
        LowerBoundInclusive,
        Unique,
        UpperBound,
        UpperBoundInclusive,
        Optional,
    )
    import numpy as np

    needed_features = request.features.union(request.entities)
    schema = {feature.name: feature.dtype.polars_type for feature in needed_features}

    exprs = {}

    for feature in needed_features:
        dtype = feature.dtype

        choices: list[Any] | None = None
        max_value: float | None = None
        min_value: float | None = None

        is_optional = False
        is_unique = False

        for constraints in feature.constraints or set():
            if isinstance(constraints, InDomain):
                choices = constraints.values
            elif isinstance(constraints, LowerBound):
                min_value = constraints.value
            elif isinstance(constraints, LowerBoundInclusive):
                min_value = constraints.value
            elif isinstance(constraints, UpperBound):
                max_value = constraints.value
            elif isinstance(constraints, UpperBoundInclusive):
                max_value = constraints.value
            elif isinstance(constraints, Unique):
                is_unique = True
            elif isinstance(constraints, Optional):
                is_optional = True

        if dtype == FeatureType.bool():
            values = np.random.choice([True, False], size=size)
        elif dtype.is_numeric:
            if is_unique:
                values = np.arange(0, size, dtype=dtype.pandas_type)
            else:
                values = np.random.random(size) * 1000

                if max_value is not None:
                    values = values * max_value

            if min_value is not None:
                values = values - min_value
        elif dtype.is_datetime:
            values = [
                datetime.now(tz=timezone.utc) - np.random.random() * timedelta(days=365) for _ in range(size)
            ]
        elif dtype.is_embedding:
            embedding_size = dtype.embedding_size() or 10
            values = np.random.random((size, embedding_size))
        else:
            if choices:
                values = np.random.choice(choices, size=size)
            else:
                values = np.random.choice(list('abcde'), size=size)

        if is_optional:
            values = np.where(np.random.random(size) > 0.5, values, np.NaN)

        exprs[feature.name] = values

    return pl.DataFrame(exprs, schema=schema)


class DummyDataSource(BatchDataSource):
    """
    The DummyDataBatchSource is a data source that generates random data for a given request.
    This can be useful for testing and development purposes.

    It will use the data types and constraints defined on a feature to generate the data.

    ```python
    from aligned import feature_view, Int64, String, DummyDataBatchSource

    @feature_view(
        source=DummyDataBatchSource(),
    )
    class MyView:
        passenger_id = Int64().as_entity()
        survived = Bool()
        age = Float().lower_bound(0).upper_bound(100)
        name = String()
        sex = String().accepted_values(["male", "female"])
    ```
    """

    type_name: str = 'dummy_data'

    def job_group_key(self) -> str:
        return self.type_name

    @classmethod
    def multi_source_features_for(
        cls: type[T], facts: RetrivalJob, requests: list[tuple[T, RetrivalRequest]]
    ) -> RetrivalJob:
        async def random_features_for(facts: RetrivalJob, request: RetrivalRequest) -> pl.LazyFrame:
            df = await facts.to_polars()
            random = data_for_request(request, df.height).lazy()
            join_columns = set(request.all_returned_columns) - set(df.columns)
            return df.hstack(random.select(pl.col(join_columns)).collect()).lazy()

        request = RetrivalRequest.unsafe_combine([request for _, request in requests])
        return CustomMethodDataSource.from_methods(
            features_for=random_features_for,
        ).features_for(facts, request)

    def all_data(self, request: RetrivalRequest, limit: int | None = None) -> RetrivalJob:
        from aligned import CustomMethodDataSource

        async def all_data(request: RetrivalRequest, limit: int | None = None) -> pl.LazyFrame:
            return data_for_request(request, limit or 100).lazy()

        return CustomMethodDataSource.from_methods(all_data=all_data).all_data(request, limit)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:
        from aligned import CustomMethodDataSource

        async def between_date(
            request: RetrivalRequest, start_date: datetime, end_date: datetime
        ) -> pl.LazyFrame:
            return data_for_request(request, 100).lazy()

        return CustomMethodDataSource.from_methods(all_between_dates=between_date).all_between_dates(
            request, start_date, end_date
        )

    async def schema(self) -> dict[str, FeatureType]:
        return {}

    def depends_on(self) -> set[FeatureLocation]:
        return set()
