from abc import ABC, abstractmethod
from aladdin.feature import Feature
from aladdin.codable import Codable
from mashumaro.types import SerializableType
from typing import Optional

class BatchDataSourceFactory:

    supported_data_sources: dict[str, type["BatchDataSource"]]

    _shared: Optional["BatchDataSourceFactory"] = None

    def __init__(self):
        from aladdin.psql.data_source import PostgreSQLDataSource
        self.supported_data_sources = {
            PostgreSQLDataSource.type_name: PostgreSQLDataSource,
        }

    @classmethod
    def shared(cls) -> "BatchDataSourceFactory":
        if cls._shared:
            return cls._shared
        cls._shared = BatchDataSourceFactory()
        return cls._shared


class BatchDataSource(ABC, Codable, SerializableType):
    """
    A source of chunked data. This could be a file, sql query, database table, etc.
    """
    event_timestamp_column: str
    
    type_name: str

    @abstractmethod
    def job_group_key(self) -> str:
        pass

    def _serialize(self):
        return self.to_dict()
    
    @classmethod
    def _deserialize(cls, value: dict[str]) -> 'BatchDataSource':
        name_type = value["type_name"]
        del value["type_name"]
        data_class = BatchDataSourceFactory.shared().supported_data_sources[name_type]
        return data_class.from_dict(value)


class ColumnFeatureMappable:
    column_feature_map: dict[str, str]

    def columns_for(self, features: list[Feature]) -> list[str]:
        return [self.column_feature_map.get(feature.name, feature.name) for feature in features]

    def feature_identifier_for(self, columns: list[str]) -> list[str]:
        reverse_map = {v: k for k, v in self.column_feature_map.items()}
        return [reverse_map.get(column, column) for column in columns]