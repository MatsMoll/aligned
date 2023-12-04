from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Any
from dataclasses import dataclass

from mashumaro.types import SerializableType
from aligned.data_file import DataFileReference

from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import EventTimestamp, Feature, FeatureLocation
from aligned.request.retrival_request import RetrivalRequest
from aligned.compiler.feature_factory import FeatureFactory

if TYPE_CHECKING:
    from aligned.retrival_job import RetrivalJob
    from datetime import datetime


class BatchDataSourceFactory:

    supported_data_sources: dict[str, type[BatchDataSource]]

    _shared: BatchDataSourceFactory | None = None

    def __init__(self) -> None:
        from aligned.sources.local import CsvFileSource, ParquetFileSource, DeltaFileSource
        from aligned.sources.psql import PostgreSQLDataSource
        from aligned.sources.redshift import RedshiftSQLDataSource
        from aligned.sources.s3 import AwsS3CsvDataSource, AwsS3ParquetDataSource
        from aligned.schemas.feature_view import FeatureViewReferenceSource

        source_types = [
            PostgreSQLDataSource,
            ParquetFileSource,
            CsvFileSource,
            DeltaFileSource,
            AwsS3CsvDataSource,
            AwsS3ParquetDataSource,
            RedshiftSQLDataSource,
            JoinDataSource,
            JoinAsofDataSource,
            FilteredDataSource,
            FeatureViewReferenceSource,
        ]

        self.supported_data_sources = {source.type_name: source for source in source_types}

    @classmethod
    def shared(cls) -> BatchDataSourceFactory:
        if cls._shared:
            return cls._shared
        cls._shared = BatchDataSourceFactory()
        return cls._shared


T = TypeVar('T')


class BatchSourceModification:

    source: BatchDataSource

    def wrap_job(self, job: RetrivalJob) -> RetrivalJob:
        raise NotImplementedError()


class BatchDataSource(ABC, Codable, SerializableType):
    """
    A definition to where a specific pice of data can be found.
    E.g: A database table, a file, a web service, etc.

    Ths can thereafter be combined with other BatchDataSources in order to create a rich dataset.
    """

    type_name: str

    @abstractmethod
    def job_group_key(self) -> str:
        """
        A key defining which sources can be grouped together in one request.
        """
        pass

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

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:
        if isinstance(self, BatchSourceModification):
            return self.wrap_job(self.source.all_data(request, limit))

        if isinstance(self, DataFileReference):
            from aligned.local.job import FileFullJob

            return FileFullJob(self, request=request, limit=limit)

        raise NotImplementedError()

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

        raise NotImplementedError()

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
            return source.wrap_job(type(source.source).multi_source_features_for(facts, requests))
        elif isinstance(source, DataFileReference):
            from aligned.local.job import FileFactualJob

            return FileFactualJob(source, [request for _, request in requests], facts)
        else:
            raise NotImplementedError(f'Type: {cls} have not implemented how to load fact data')

    def features_for(self, facts: RetrivalJob, request: RetrivalRequest) -> RetrivalJob:
        return type(self).multi_source_features_for(facts, [(self, request)])

    async def schema(self) -> dict[str, FeatureFactory]:
        """Returns the schema for the data source

        ```python
        source = FileSource.parquet_at('test_data/titanic.parquet')
        schema = await source.schema()
        >>> {'passenger_id': FeatureType(name='int64'), ...}
        ```

        Raises:
            NotImplementedError: By default will this error be raised if not implemented

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
        return FeatureView.feature_view_code_template(schema, f'{self}', view_name)

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

    def depends_on(self) -> set[FeatureLocation]:
        return set()


@dataclass
class FilteredDataSource(BatchDataSource):

    source: BatchDataSource
    condition: DerivedFeature | Feature

    type_name: str = 'subset'

    def job_group_key(self) -> str:
        return f'subset/{self.source.job_group_key()}'

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:

        if isinstance(self.condition, Feature):
            request.features.add(self.condition)
        else:
            request.derived_features.add(self.condition)

        return self.source.all_data(request, limit).filter(self.condition).derive_features([request])

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


def view_wrapper_instance_source(view: Any) -> tuple[BatchDataSource, RetrivalRequest]:
    from aligned.feature_view.feature_view import FeatureViewWrapper
    from aligned.schemas.feature_view import FeatureViewReferenceSource

    if not hasattr(view, '__view_wrapper__'):
        raise ValueError(
            f'Unable to join {view} as a __view_wrapper__ is needed. Make sure you have used @feature_view'
        )

    wrapper = getattr(view, '__view_wrapper__')
    if not isinstance(wrapper, FeatureViewWrapper):
        raise ValueError()

    compiled_view = wrapper.compile()

    return (FeatureViewReferenceSource(compiled_view), compiled_view.request_all.needed_requests[0])


def join_asof_source(
    source: BatchDataSource,
    left_request: RetrivalRequest,
    view: Any,
    left_on: list[str] | None = None,
    right_on: list[str] | None = None,
) -> JoinAsofDataSource:

    right_source, right_request = view_wrapper_instance_source(view)

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
    on: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
    how: str = 'inner',
    left_request: RetrivalRequest | None = None,
) -> JoinDataSource:
    from aligned.data_source.batch_data_source import JoinDataSource
    from aligned.feature_view.feature_view import FeatureViewWrapper

    right_source, right_request = view_wrapper_instance_source(view)

    if on is None:
        on_keys = list(right_request.entity_names)
    else:
        on_keys = resolve_keys(on)

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
        left_on=on_keys,
        right_on=on_keys,
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

    type_name: str = 'join_asof'

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
            )
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
            )
            .derive_features([request])
        )

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
        return self.source.depends_on().intersection(self.right_source.depends_on())


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
        )

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:

        right_job = self.right_source.all_data(self.right_request, limit=None).derive_features(
            [self.right_request]
        )

        return (
            self.source.all_data(self.left_request, limit=limit)
            .derive_features([self.left_request])
            .join(right_job, method=self.method, left_on=self.left_on, right_on=self.right_on)
            .derive_features([request])
        )

    def join(
        self,
        view: Any,
        on: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        how: str = 'inner',
    ) -> BatchDataSource:
        return join_source(self, view, on, how)

    def depends_on(self) -> set[FeatureLocation]:
        return self.source.depends_on().intersection(self.right_source.depends_on())


class ColumnFeatureMappable:
    mapping_keys: dict[str, str]

    def with_renames(self: T, mapping_keys: dict[str, str]) -> T:
        self.mapping_keys = mapping_keys  # type: ignore
        return self

    def columns_for(self, features: list[Feature]) -> list[str]:
        return [self.mapping_keys.get(feature.name, feature.name) for feature in features]

    def feature_identifier_for(self, columns: list[str]) -> list[str]:
        reverse_map = {v: k for k, v in self.mapping_keys.items()}
        return [reverse_map.get(column, column) for column in columns]
