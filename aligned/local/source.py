from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

import pandas as pd
from httpx import HTTPStatusError

from aligned.data_file import DataFileReference
from aligned.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aligned.enricher import CsvFileEnricher, Enricher, LoadedStatEnricher, StatisticEricher, TimespanSelector
from aligned.exceptions import UnableToFindFileException
from aligned.feature_store import FeatureStore
from aligned.local.job import FileDateJob, FileFactualJob, FileFullJob
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob
from aligned.s3.storage import FileStorage, HttpStorage
from aligned.schemas.codable import Codable
from aligned.schemas.repo_definition import RepoDefinition
from aligned.storage import Storage

logger = logging.getLogger(__name__)


class StorageFileReference:
    """
    A reference to a file that can be loaded as bytes.

    The bytes can contain anything, potentially a FeatureStore definition
    """

    async def read(self) -> bytes:
        raise NotImplementedError()

    async def write(self, content: bytes) -> None:
        raise NotImplementedError()

    async def feature_store(self) -> FeatureStore:
        file = await self.read()
        return FeatureStore.from_definition(RepoDefinition.from_json(file))


@dataclass
class CsvConfig(Codable):
    """
    A config for how a CSV file should be loaded
    """

    seperator: str = field(default=',')
    compression: Literal['infer', 'gzip', 'bz2', 'zip', 'xz', 'zstd'] = field(default='infer')
    should_write_index: bool = field(default=False)


@dataclass
class CsvFileSource(BatchDataSource, ColumnFeatureMappable, StatisticEricher, DataFileReference):
    """
    A source pointing to a CSV file
    """

    path: str
    mapping_keys: dict[str, str] = field(default_factory=dict)
    csv_config: CsvConfig = field(default_factory=CsvConfig)

    type_name: str = 'csv'

    def job_group_key(self) -> str:
        return f'{self.type_name}/{self.path}'

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    async def read_pandas(self) -> pd.DataFrame:
        try:
            return pd.read_csv(
                self.path, sep=self.csv_config.seperator, compression=self.csv_config.compression
            )
        except FileNotFoundError:
            raise UnableToFindFileException()
        except HTTPStatusError:
            raise UnableToFindFileException()

    async def write_pandas(self, df: pd.DataFrame) -> None:
        df.to_csv(
            self.path,
            sep=self.csv_config.seperator,
            compression=self.csv_config.compression,
            index=self.csv_config.should_write_index,
        )

    def std(
        self, columns: set[str], time: TimespanSelector | None = None, limit: int | None = None
    ) -> Enricher:
        return LoadedStatEnricher(
            stat='std',
            columns=list(columns),
            enricher=self.enricher().selector(time, limit),
            mapping_keys=self.mapping_keys,
        )

    def mean(
        self, columns: set[str], time: TimespanSelector | None = None, limit: int | None = None
    ) -> Enricher:
        return LoadedStatEnricher(
            stat='mean',
            columns=list(columns),
            enricher=self.enricher().selector(time, limit),
            mapping_keys=self.mapping_keys,
        )

    def enricher(self) -> CsvFileEnricher:
        return CsvFileEnricher(file=self.path)

    def all_data(self, request: RetrivalRequest, limit: int | None) -> FullExtractJob:
        return FileFullJob(self, request, limit)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> DateRangeJob:
        return FileDateJob(source=self, request=request, start_date=start_date, end_date=end_date)

    @classmethod
    def feature_for(
        cls, facts: dict[str, list], requests: dict[CsvFileSource, RetrivalRequest]
    ) -> FactualRetrivalJob:
        if len(requests.keys()) != 1:
            raise ValueError(f'Only able to load one {requests} at a time')

        source = list(requests.keys())[0]
        if not isinstance(source, cls):
            raise ValueError(f'Only {cls} is supported, recived: {source}')

        # Group based on config
        return FileFactualJob(
            source=source,
            requests=list(requests.values()),
            facts=facts,
        )


@dataclass
class ParquetConfig(Codable):
    """
    A config for how a CSV file should be loaded
    """

    engine: Literal['auto', 'pyarrow', 'fastparquet'] = field(default='auto')
    compression: Literal['snappy', 'gzip', 'brotli', None] = field(default='snappy')
    should_write_index: bool = field(default=False)


@dataclass
class ParquetFileSource(BatchDataSource, ColumnFeatureMappable):
    """
    A source pointing to a Parquet file
    """

    path: str
    mapping_keys: dict[str, str] = field(default_factory=dict)
    config: ParquetConfig = field(default_factory=ParquetConfig)

    type_name: str = 'parquet'

    def job_group_key(self) -> str:
        return f'{self.type_name}/{self.path}'

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    async def read_pandas(self) -> pd.DataFrame:
        try:
            return pd.read_parquet(self.path)
        except FileNotFoundError:
            raise UnableToFindFileException()
        except HTTPStatusError:
            raise UnableToFindFileException()

    async def write_pandas(self, df: pd.DataFrame) -> None:
        df.to_parquet(
            self.path,
            engine=self.config.engine,
            compression=self.config.compression,
            index=self.config.should_write_index,
        )

    def all_data(self, request: RetrivalRequest, limit: int | None) -> FullExtractJob:
        return FileFullJob(self, request, limit)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> DateRangeJob:
        return FileDateJob(source=self, request=request, start_date=start_date, end_date=end_date)

    @classmethod
    def feature_for(
        cls, facts: dict[str, list], requests: dict[ParquetFileSource, RetrivalRequest]
    ) -> FactualRetrivalJob:
        if len(requests.keys()) != 1:
            raise ValueError(f'Only able to load one {requests} at a time')

        source = list(requests.keys())[0]
        if not isinstance(source, cls):
            raise ValueError(f'Only {cls} is supported, recived: {source}')

        # Group based on config
        return FileFactualJob(
            source=source,
            requests=list(requests.values()),
            facts=facts,
        )


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
    """
    A factory class, creating references to files.

    This therefore abstracts away the concrete classes the users wants.
    Therefore making them easier to discover.
    """

    @staticmethod
    def from_path(path: str) -> StorageFileSource:
        return StorageFileSource(path=path)

    @staticmethod
    def csv_at(
        path: str, mapping_keys: dict[str, str] | None = None, csv_config: CsvConfig | None = None
    ) -> CsvFileSource:
        return CsvFileSource(path, mapping_keys=mapping_keys or {}, csv_config=csv_config or CsvConfig())

    @staticmethod
    def parquet_at(
        path: str, mapping_keys: dict[str, str] | None = None, config: ParquetConfig | None = None
    ) -> ParquetFileSource:
        return ParquetFileSource(path=path, mapping_keys=mapping_keys or {}, config=config or ParquetConfig())


@dataclass
class LiteralReference(DataFileReference):
    """
    A class containing a in mem pandas frame.

    This makes it easier standardise the interface when writing data.
    """

    file: pd.DataFrame

    def job_group_key(self) -> str:
        return str(uuid4())

    async def read_pandas(self) -> pd.DataFrame:
        return self.file
