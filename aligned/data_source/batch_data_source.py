from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Any
from dataclasses import dataclass

from mashumaro.types import SerializableType
from aligned.data_file import DataFileReference

from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import EventTimestamp, Feature

if TYPE_CHECKING:
    from aligned.compiler.feature_factory import FeatureFactory
    from aligned.request.retrival_request import RetrivalRequest
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

        source_types = [
            PostgreSQLDataSource,
            ParquetFileSource,
            CsvFileSource,
            DeltaFileSource,
            AwsS3CsvDataSource,
            AwsS3ParquetDataSource,
            RedshiftSQLDataSource,
            JoinDataSource,
            FilteredDataSource,
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
        from aligned import FeatureView

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


@dataclass
class FilteredDataSource(BatchSourceModification, BatchDataSource):

    source: BatchDataSource
    condition: DerivedFeature | Feature

    type_name: str = 'subset'

    def job_group_key(self) -> str:
        return f'subset/{self.source.job_group_key()}'

    def wrap_job(self, job: RetrivalJob) -> RetrivalJob:
        return job.filter(self.condition)


@dataclass
class JoinDataSource(BatchSourceModification, BatchDataSource):

    source: BatchDataSource
    right_source: BatchDataSource
    right_request: RetrivalRequest
    left_on: str
    right_on: str
    method: str

    type_name: str = 'join'

    def job_group_key(self) -> str:
        return f'join/{self.source.job_group_key()}'

    def wrap_job(self, job: RetrivalJob) -> RetrivalJob:

        right_job = self.right_source.all_data(self.right_request, limit=None)
        return job.join(right_job, self.method, (self.left_on, self.right_on))


class ColumnFeatureMappable:
    mapping_keys: dict[str, str]

    def columns_for(self, features: list[Feature]) -> list[str]:
        return [self.mapping_keys.get(feature.name, feature.name) for feature in features]

    def feature_identifier_for(self, columns: list[str]) -> list[str]:
        reverse_map = {v: k for k, v in self.mapping_keys.items()}
        return [reverse_map.get(column, column) for column in columns]
