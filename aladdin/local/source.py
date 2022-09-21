import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd

from aladdin.codable import Codable
from aladdin.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aladdin.enricher import CsvFileEnricher, Enricher, LoadedStatEnricher, StatisticEricher
from aladdin.feature_store import FeatureStore
from aladdin.repo_definition import RepoDefinition
from aladdin.s3.storage import FileStorage, HttpStorage
from aladdin.storage import Storage

logger = logging.getLogger(__name__)


class DataFileReference:
    async def read_pandas(self) -> pd.DataFrame:
        raise NotImplementedError()

    async def read_dask(self) -> pd.DataFrame:
        raise NotImplementedError()

    async def write_pandas(self) -> None:
        raise NotImplementedError()

    async def write_dask(self) -> None:
        raise NotImplementedError()


class StorageFileReference:
    async def read(self) -> bytes:
        raise NotImplementedError()

    async def write(self, content: bytes) -> None:
        raise NotImplementedError()

    async def feature_store(self) -> FeatureStore:
        file = await self.read()
        return FeatureStore.from_definition(RepoDefinition.from_json(file))


@dataclass
class CsvConfig(Codable):
    seperator: str = field(default=',')
    compression: Literal['infer', 'gzip', 'bz2', 'zip', 'xz', 'zstd'] = field(default='infer')


@dataclass
class CsvFileSource(BatchDataSource, ColumnFeatureMappable, StatisticEricher):

    path: str
    mapping_keys: dict[str, str] = field(default_factory=dict)
    csv_config: CsvConfig = field(default_factory=CsvConfig())

    type_name: str = 'csv'

    def job_group_key(self) -> str:
        return f'{self.type_name}/{self.path}'

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    async def read_pandas(self) -> pd.DataFrame:
        return pd.read_csv(self.path, sep=self.csv_config.seperator, compression=self.csv_config.compression)

    def std(self, columns: set[str]) -> Enricher:
        return LoadedStatEnricher(stat='std', columns=list(columns), enricher=self.enricher())

    def mean(self, columns: set[str]) -> Enricher:
        return LoadedStatEnricher(stat='mean', columns=list(columns), enricher=self.enricher())

    def enricher(self) -> Enricher:
        return CsvFileEnricher(file=Path(self.path))


@dataclass
class StorageFileSource(StorageFileReference):

    path: str

    @property
    def storage(self) -> Storage:
        if self.path.startswith('http'):
            return HttpStorage()
        else:
            return FileStorage()

    def __hash__(self) -> int:
        return hash(self.path)

    async def read(self) -> bytes:
        return await self.storage.read(self.path)

    async def write(self, content: bytes) -> None:
        return await self.storage.write(self.path, content)


class FileSource:
    @staticmethod
    def from_path(path: str) -> 'StorageFileSource':
        return StorageFileSource(path=path)

    @staticmethod
    def csv_at(
        path: str, mapping_keys: dict[str, str] | None = None, csv_config: CsvConfig | None = None
    ) -> 'CsvFileSource':
        return CsvFileSource(path, mapping_keys=mapping_keys or {}, csv_config=csv_config or CsvConfig())


@dataclass
class LiteralReference(DataFileReference):

    file: pd.DataFrame

    def job_group_key(self) -> str:
        return str(uuid4())

    async def read_pandas(self) -> pd.DataFrame:
        return self.file
