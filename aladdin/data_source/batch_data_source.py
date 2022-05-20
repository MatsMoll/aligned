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
        from aladdin.local.source import LocalFileSource
        from aladdin.s3.config import AwsS3DataSource
        self.supported_data_sources = {
            PostgreSQLDataSource.type_name: PostgreSQLDataSource,
            LocalFileSource.type_name: LocalFileSource,
            AwsS3DataSource.type_name: AwsS3DataSource
        }

    @classmethod
    def shared(cls) -> "BatchDataSourceFactory":
        if cls._shared:
            return cls._shared
        cls._shared = BatchDataSourceFactory()
        return cls._shared


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

    def _serialize(self):
        return self.to_dict()

    def __hash__(self) -> int:
        return hash(self.job_group_key())
    
    @classmethod
    def _deserialize(cls, value: dict[str]) -> 'BatchDataSource':
        name_type = value["type_name"]
        if name_type not in BatchDataSourceFactory.shared().supported_data_sources:
            raise ValueError(f"Unknown batch data source id: '{name_type}'.\nRemember to add the data source to the BatchDataSourceFactory.supported_data_sources if it is a custom type.")
        del value["type_name"]
        data_class = BatchDataSourceFactory.shared().supported_data_sources[name_type]
        return data_class.from_dict(value)


class ColumnFeatureMappable:
    mapping_keys: dict[str, str]

    def columns_for(self, features: list[Feature]) -> list[str]:
        return [self.mapping_keys.get(feature.name, feature.name) for feature in features]

    def feature_identifier_for(self, columns: list[str]) -> list[str]:
        reverse_map = {v: k for k, v in self.mapping_keys.items()}
        return [reverse_map.get(column, column) for column in columns]