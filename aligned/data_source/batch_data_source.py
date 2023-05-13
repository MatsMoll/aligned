from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, TypeVar

from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable
from aligned.schemas.feature import Feature

if TYPE_CHECKING:
    from aligned.request.retrival_request import RetrivalRequest
    from aligned.retrival_job import DateRangeJob, FullExtractJob, RetrivalJob


class BatchDataSourceFactory:

    supported_data_sources: dict[str, type[BatchDataSource]]

    _shared: BatchDataSourceFactory | None = None

    def __init__(self) -> None:
        from aligned.sources.local import CsvFileSource, ParquetFileSource
        from aligned.sources.psql import PostgreSQLDataSource
        from aligned.sources.redshift import RedshiftSQLDataSource
        from aligned.sources.s3 import AwsS3CsvDataSource, AwsS3ParquetDataSource

        self.supported_data_sources = {
            PostgreSQLDataSource.type_name: PostgreSQLDataSource,
            ParquetFileSource.type_name: ParquetFileSource,
            CsvFileSource.type_name: CsvFileSource,
            AwsS3CsvDataSource.type_name: AwsS3CsvDataSource,
            AwsS3ParquetDataSource.type_name: AwsS3ParquetDataSource,
            RedshiftSQLDataSource.type_name: RedshiftSQLDataSource,
        }

    @classmethod
    def shared(cls) -> BatchDataSourceFactory:
        if cls._shared:
            return cls._shared
        cls._shared = BatchDataSourceFactory()
        return cls._shared


T = TypeVar('T')


class BatchDataSource(ABC, Codable, SerializableType):
    """
    A definition to where a specific pice of data can be found.
    E.g: A database table, a file, a web service, etc.

    Ths can thereafter be combined with other BatchDataSources in order to create a rich dataset.
    """

    type_name: str

    @abstractmethod
    def job_group_key(self) -> str:
        pass

    def _serialize(self) -> dict:
        assert (
            self.type_name in BatchDataSourceFactory.shared().supported_data_sources
        ), f'Unknown type_name: {self.type_name}'
        return self.to_dict()

    def __hash__(self) -> int:
        return hash(self.job_group_key())

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

    def all_data(self, request: RetrivalRequest, limit: int | None) -> FullExtractJob:
        raise NotImplementedError()

    def all_between_dates(
        self,
        request: RetrivalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> DateRangeJob:
        raise NotImplementedError()

    @classmethod
    def multi_source_features_for(
        cls: type[T], facts: RetrivalJob, requests: list[tuple[T, RetrivalRequest]]
    ) -> RetrivalJob:
        raise NotImplementedError()

    def features_for(self, facts: RetrivalJob, request: RetrivalRequest) -> RetrivalJob:
        return type(self).multi_source_features_for(facts, [(self, request)])


class ColumnFeatureMappable:
    mapping_keys: dict[str, str]

    def columns_for(self, features: list[Feature]) -> list[str]:
        return [self.mapping_keys.get(feature.name, feature.name) for feature in features]

    def feature_identifier_for(self, columns: list[str]) -> list[str]:
        reverse_map = {v: k for k, v in self.mapping_keys.items()}
        return [reverse_map.get(column, column) for column in columns]
