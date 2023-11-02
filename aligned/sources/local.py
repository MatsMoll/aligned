from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import pandas as pd
import polars as pl
from httpx import HTTPStatusError

from aligned.data_file import DataFileReference
from aligned.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aligned.enricher import CsvFileEnricher, Enricher, LoadedStatEnricher, StatisticEricher, TimespanSelector
from aligned.exceptions import UnableToFindFileException
from aligned.local.job import FileDateJob, FileFactualJob, FileFullJob
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob, RetrivalJob
from aligned.s3.storage import FileStorage, HttpStorage
from aligned.schemas.codable import Codable
from aligned.schemas.feature import EventTimestamp, FeatureType
from aligned.schemas.folder import Folder
from aligned.schemas.repo_definition import RepoDefinition
from aligned.storage import Storage
from aligned.feature_store import FeatureStore

if TYPE_CHECKING:
    from aligned.compiler.feature_factory import FeatureFactory
    from datetime import datetime


logger = logging.getLogger(__name__)


class AsRepoDefinition:
    async def as_repo_definition(self) -> RepoDefinition:
        raise NotImplementedError()

    async def feature_store(self) -> FeatureStore:
        return FeatureStore.from_definition(await self.as_repo_definition())


class StorageFileReference(AsRepoDefinition):
    """
    A reference to a file that can be loaded as bytes.

    The bytes can contain anything, potentially a FeatureStore definition
    """

    async def read(self) -> bytes:
        raise NotImplementedError()

    async def write(self, content: bytes) -> None:
        raise NotImplementedError()

    async def as_repo_definition(self) -> RepoDefinition:
        file = await self.read()
        return RepoDefinition.from_json(file)


async def data_file_freshness(reference: DataFileReference, column_name: str) -> datetime | None:
    try:
        file = await reference.to_polars()
        return file.select(column_name).max().collect()[0, column_name]
    except UnableToFindFileException:
        return None


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

    async def to_polars(self) -> pl.LazyFrame:

        if self.path.startswith('http'):
            from io import BytesIO

            buffer = await HttpStorage().read(self.path)
            io_buffer = BytesIO(buffer)
            io_buffer.seek(0)
            return pl.read_csv(io_buffer, separator=self.csv_config.seperator, try_parse_dates=True).lazy()

        return pl.scan_csv(self.path, separator=self.csv_config.seperator, try_parse_dates=True)

    async def write_pandas(self, df: pd.DataFrame) -> None:
        df.to_csv(
            self.path,
            sep=self.csv_config.seperator,
            compression=self.csv_config.compression,
            index=self.csv_config.should_write_index,
        )

    async def write_polars(self, df: pl.LazyFrame) -> None:
        await self.write_pandas(df.collect().to_pandas())

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
    def multi_source_features_for(
        cls, facts: RetrivalJob, requests: list[tuple[CsvFileSource, RetrivalRequest]]
    ) -> FactualRetrivalJob:
        sources = {source for source, _ in requests}
        if len(sources) != 1:
            raise ValueError(f'Only able to load one {requests} at a time')

        source = list(sources)[0]
        if not isinstance(source, cls):
            raise ValueError(f'Only {cls} is supported, recived: {source}')

        # Group based on config
        return FileFactualJob(
            source=source,
            requests=[request for _, request in requests],
            facts=facts,
        )

    async def schema(self) -> dict[str, FeatureFactory]:
        df = await self.to_polars()
        return {name: FeatureType.from_polars(pl_type).feature_factory for name, pl_type in df.schema.items()}

    async def feature_view_code(self, view_name: str) -> str:
        from aligned import FeatureView

        schema = await self.schema()
        data_source_code = f'FileSource.csv_at("{self.path}", csv_config={self.csv_config})'
        return FeatureView.feature_view_code_template(
            schema,
            data_source_code,
            view_name,
            'from aligned import FileSource\nfrom aligned.sources.local import CsvConfig',
        )

    async def freshness(self, event_timestamp: EventTimestamp) -> datetime | None:
        return await data_file_freshness(self, event_timestamp.name)


@dataclass
class ParquetConfig(Codable):
    """
    A config for how a CSV file should be loaded
    """

    engine: Literal['auto', 'pyarrow', 'fastparquet'] = field(default='auto')
    compression: Literal['snappy', 'gzip', 'brotli', None] = field(default='snappy')
    should_write_index: bool = field(default=False)


@dataclass
class ParquetFileSource(BatchDataSource, ColumnFeatureMappable, DataFileReference):
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

    async def to_polars(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.path)

    async def write_polars(self, df: pl.LazyFrame) -> None:
        df.collect().write_parquet(self.path, compression=self.config.compression)

    def all_data(self, request: RetrivalRequest, limit: int | None) -> FullExtractJob:
        return FileFullJob(self, request, limit)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> DateRangeJob:
        return FileDateJob(source=self, request=request, start_date=start_date, end_date=end_date)

    @classmethod
    def multi_source_features_for(
        cls, facts: RetrivalJob, requests: list[tuple[ParquetFileSource, RetrivalRequest]]
    ) -> FactualRetrivalJob:

        source = requests[0][0]
        if not isinstance(source, cls):
            raise ValueError(f'Only {cls} is supported, recived: {source}')

        # Group based on config
        return FileFactualJob(
            source=source,
            requests=[request for _, request in requests],
            facts=facts,
        )

    async def schema(self) -> dict[str, FeatureFactory]:
        parquet_schema = pl.read_parquet_schema(self.path)
        return {
            name: FeatureType.from_polars(pl_type).feature_factory for name, pl_type in parquet_schema.items()
        }

    async def feature_view_code(self, view_name: str) -> str:
        from aligned import FeatureView

        schema = await self.schema()
        data_source_code = f'FileSource.parquet_at("{self.path}")'
        return FeatureView.feature_view_code_template(
            schema, data_source_code, view_name, 'from aligned import FileSource'
        )

    async def freshness(self, event_timestamp: EventTimestamp) -> datetime | None:
        df = await self.to_polars()
        et_name = event_timestamp.name
        return df.select(et_name).max().collect()[0, et_name]


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
        await self.storage.write(self.path, content)


@dataclass
class DirectoryRepo(AsRepoDefinition):

    dir: Path
    exclude: list[str] | None = field(default=None)

    async def as_repo_definition(self) -> RepoDefinition:
        from aligned.compiler.repo_reader import RepoReader

        return await RepoReader.definition_from_path(self.dir, self.exclude)


class FileSource:
    """
    A factory class, creating references to files.

    This therefore abstracts away the concrete classes the users wants.
    Therefore making them easier to discover.
    """

    @staticmethod
    def json_at(path: str) -> StorageFileSource:
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

    @staticmethod
    def folder(path: str) -> Folder:
        return LocalFolder(base_path=Path(path))

    @staticmethod
    def repo_from_dir(dir: str, exclude: list[str] | None = None) -> AsRepoDefinition:

        return DirectoryRepo(Path(dir), exclude)


@dataclass
class LocalFolder(Folder):

    base_path: Path
    name = 'local_folder'

    def file_at(self, path: Path) -> StorageFileReference:
        return StorageFileSource(path=str(self.base_path / path))


class LiteralReference(DataFileReference):
    """
    A class containing a in mem pandas frame.

    This makes it easier standardise the interface when writing data.
    """

    file: pl.LazyFrame

    def __init__(self, file: pl.LazyFrame | pd.DataFrame) -> None:
        if isinstance(file, pd.DataFrame):
            self.file = pl.from_pandas(file).lazy()
        else:
            self.file = file

    def job_group_key(self) -> str:
        return str(uuid4())

    async def read_pandas(self) -> pd.DataFrame:
        return self.file.collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        return self.file
