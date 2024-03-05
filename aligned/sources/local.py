from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import pandas as pd
import polars as pl
from httpx import HTTPStatusError

from aligned.data_file import DataFileReference, upsert_on_column
from aligned.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aligned.enricher import CsvFileEnricher, Enricher, LoadedStatEnricher, TimespanSelector
from aligned.exceptions import UnableToFindFileException
from aligned.local.job import FileDateJob, FileFactualJob, FileFullJob
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import RetrivalJob
from aligned.s3.storage import FileStorage, HttpStorage
from aligned.schemas.codable import Codable
from aligned.schemas.feature import EventTimestamp, FeatureType
from aligned.storage import Storage
from aligned.feature_source import WritableFeatureSource
from aligned.schemas.date_formatter import DateFormatter

if TYPE_CHECKING:
    from aligned.compiler.feature_factory import FeatureFactory
    from datetime import datetime
    from aligned.schemas.repo_definition import RepoDefinition
    from aligned.feature_store import FeatureStore


logger = logging.getLogger(__name__)


class AsRepoDefinition:
    async def as_repo_definition(self) -> RepoDefinition:
        raise NotImplementedError()

    async def feature_store(self) -> FeatureStore:
        from aligned.feature_store import FeatureStore

        return FeatureStore.from_definition(await self.as_repo_definition())


class StorageFileReference(AsRepoDefinition):
    """
    A reference to a file that can be loaded as bytes.

    The bytes can contain anything, potentially a FeatureStore definition
    """

    async def read(self) -> bytes:
        raise NotImplementedError(type(self))

    async def write(self, content: bytes) -> None:
        raise NotImplementedError(type(self))

    async def as_repo_definition(self) -> RepoDefinition:
        from aligned.schemas.repo_definition import RepoDefinition

        file = await self.read()
        return RepoDefinition.from_json(file)


async def data_file_freshness(reference: DataFileReference, column_name: str) -> datetime | None:
    try:
        file = await reference.to_lazy_polars()
        if isinstance(reference, ColumnFeatureMappable):
            source_column = reference.feature_identifier_for([column_name])[0]
        else:
            source_column = column_name

        return file.select(source_column).max().collect()[0, source_column]
    except UnableToFindFileException:
        return None


def create_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(exist_ok=True)


def do_file_exist(path: str) -> bool:
    return Path(path).is_file()


@dataclass
class CsvConfig(Codable):
    """
    A config for how a CSV file should be loaded
    """

    seperator: str = field(default=',')
    compression: Literal['infer', 'gzip', 'bz2', 'zip', 'xz', 'zstd'] = field(default='infer')
    should_write_index: bool = field(default=False)


@dataclass
class CsvFileSource(BatchDataSource, ColumnFeatureMappable, DataFileReference, WritableFeatureSource):
    """
    A source pointing to a CSV file
    """

    path: str
    mapping_keys: dict[str, str] = field(default_factory=dict)
    csv_config: CsvConfig = field(default_factory=CsvConfig)
    formatter: DateFormatter = field(default_factory=DateFormatter.iso_8601)

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
            raise UnableToFindFileException(self.path)
        except HTTPStatusError:
            raise UnableToFindFileException(self.path)

    async def to_lazy_polars(self) -> pl.LazyFrame:

        if self.path.startswith('http'):
            from io import BytesIO

            buffer = await HttpStorage().read(self.path)
            io_buffer = BytesIO(buffer)
            io_buffer.seek(0)
            return pl.read_csv(io_buffer, separator=self.csv_config.seperator, try_parse_dates=True).lazy()

        if not do_file_exist(self.path):
            raise UnableToFindFileException(self.path)

        try:
            return pl.scan_csv(self.path, separator=self.csv_config.seperator, try_parse_dates=True)
        except OSError:
            raise UnableToFindFileException(self.path)

    async def upsert(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:
        if len(requests) != 1:
            raise ValueError('Csv files only support one write request as of now')

        request = requests[0]

        data = (await job.to_lazy_polars()).select(request.all_returned_columns)
        potential_timestamps = request.all_features

        if request.event_timestamp:
            potential_timestamps.add(request.event_timestamp)

        for feature in potential_timestamps:
            if feature.dtype.name == 'datetime':
                data = data.with_columns(self.formatter.encode_polars(feature.name))

        if self.mapping_keys:
            columns = self.feature_identifier_for(data.columns)
            data = data.rename(dict(zip(data.columns, columns)))

        new_df = data.select(request.all_returned_columns)
        entities = list(request.entity_names)
        try:
            existing_df = await self.to_lazy_polars()
            write_df = upsert_on_column(entities, new_df, existing_df)
        except UnableToFindFileException:
            write_df = new_df

        await self.write_polars(write_df)

    async def insert(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:
        if len(requests) != 1:
            raise ValueError('Csv files only support one write request as of now')

        request = requests[0]

        data = (await job.to_lazy_polars()).select(request.all_returned_columns)
        for feature in request.features:
            if feature.dtype.name == 'datetime':
                data = data.with_columns(self.formatter.encode_polars(feature.name))

        if self.mapping_keys:
            columns = self.feature_identifier_for(data.columns)
            data = data.rename(dict(zip(data.columns, columns)))

        try:
            existing_df = await self.to_lazy_polars()
            write_df = pl.concat([data, existing_df.select(data.columns)], how='vertical_relaxed')
        except UnableToFindFileException:
            write_df = data

        await self.write_polars(write_df)

    async def write_pandas(self, df: pd.DataFrame) -> None:
        create_parent_dir(self.path)
        df.to_csv(
            self.path,
            sep=self.csv_config.seperator,
            compression=self.csv_config.compression,
            index=self.csv_config.should_write_index,
        )

    async def write_polars(self, df: pl.LazyFrame) -> None:
        create_parent_dir(self.path)
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

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:
        return FileFullJob(self, request, limit, date_formatter=self.formatter)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:
        return FileDateJob(
            source=self,
            request=request,
            start_date=start_date,
            end_date=end_date,
            date_formatter=self.formatter,
        )

    @classmethod
    def multi_source_features_for(
        cls, facts: RetrivalJob, requests: list[tuple[CsvFileSource, RetrivalRequest]]
    ) -> RetrivalJob:
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
            date_formatter=source.formatter,
        )

    async def schema(self) -> dict[str, FeatureFactory]:
        df = await self.to_lazy_polars()
        return {name: FeatureType.from_polars(pl_type).feature_factory for name, pl_type in df.schema.items()}

    async def feature_view_code(self, view_name: str) -> str:
        from aligned.feature_view.feature_view import FeatureView

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
        create_parent_dir(self.path)
        df.to_parquet(
            self.path,
            engine=self.config.engine,
            compression=self.config.compression,
            index=self.config.should_write_index,
        )

    async def to_lazy_polars(self) -> pl.LazyFrame:

        if (not self.path.startswith('http')) and (not do_file_exist(self.path)):
            raise UnableToFindFileException(self.path)

        try:
            return pl.scan_parquet(self.path)
        except OSError:
            raise UnableToFindFileException(self.path)

    async def write_polars(self, df: pl.LazyFrame) -> None:
        create_parent_dir(self.path)
        df.collect().write_parquet(self.path, compression=self.config.compression)

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:
        return FileFullJob(self, request, limit)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:
        return FileDateJob(source=self, request=request, start_date=start_date, end_date=end_date)

    @classmethod
    def multi_source_features_for(
        cls, facts: RetrivalJob, requests: list[tuple[ParquetFileSource, RetrivalRequest]]
    ) -> RetrivalJob:

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
        if self.path.startswith('http'):
            parquet_schema = pl.scan_parquet(self.path).schema
        else:
            parquet_schema = pl.read_parquet_schema(self.path)

        return {
            name: FeatureType.from_polars(pl_type).feature_factory for name, pl_type in parquet_schema.items()
        }

    async def feature_view_code(self, view_name: str) -> str:
        from aligned.feature_view.feature_view import FeatureView

        schema = await self.schema()
        data_source_code = f'FileSource.parquet_at("{self.path}")'
        return FeatureView.feature_view_code_template(
            schema, data_source_code, view_name, 'from aligned import FileSource'
        )


@dataclass
class DeltaFileConfig(Codable):

    mode: Literal['append', 'overwrite', 'error'] = field(default='append')
    overwrite_schema: bool = field(default=False)


@dataclass
class DeltaFileSource(BatchDataSource, ColumnFeatureMappable, DataFileReference, WritableFeatureSource):
    """
    A source pointing to a Parquet file
    """

    path: str
    mapping_keys: dict[str, str] = field(default_factory=dict)
    config: DeltaFileConfig = field(default_factory=DeltaFileConfig)

    type_name: str = 'delta'

    def job_group_key(self) -> str:
        return f'{self.type_name}/{self.path}'

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    async def read_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    async def write_pandas(self, df: pd.DataFrame) -> None:
        await self.write_polars(pl.from_pandas(df).lazy())

    async def to_lazy_polars(self) -> pl.LazyFrame:
        if not do_file_exist(self.path):
            raise UnableToFindFileException(self.path)

        try:
            return pl.scan_delta(self.path)
        except OSError:
            raise UnableToFindFileException(self.path)

    async def write_polars(self, df: pl.LazyFrame) -> None:
        create_parent_dir(self.path)
        df.collect().write_delta(
            self.path, mode=self.config.mode, overwrite_schema=self.config.overwrite_schema
        )

    async def schema(self) -> dict[str, FeatureFactory]:
        parquet_schema = pl.read_delta(self.path).schema
        return {
            name: FeatureType.from_polars(pl_type).feature_factory for name, pl_type in parquet_schema.items()
        }

    async def feature_view_code(self, view_name: str) -> str:
        from aligned.feature_view.feature_view import FeatureView

        schema = await self.schema()
        data_source_code = f'FileSource.parquet_at("{self.path}")'
        return FeatureView.feature_view_code_template(
            schema, data_source_code, view_name, 'from aligned import FileSource'
        )

    async def insert(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:
        if len(requests) != 1:
            raise ValueError('Delta files only support one write request as of now')

        request = requests[0]

        data = await job.to_lazy_polars()
        data.select(request.all_returned_columns).collect().write_delta(self.path, mode='append')

    async def upsert(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:
        if len(requests) != 1:
            raise ValueError('Delta files only support one write request as of now')

        request = requests[0]

        new_data = await job.to_lazy_polars()
        existing = await self.to_lazy_polars()

        upsert_on_column(list(request.entity_names), new_data, existing).collect().write_delta(
            self.path, mode='overwrite'
        )


@dataclass
class StorageFileSource(StorageFileReference, Codable):

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


@dataclass
class FileDirectory(Codable):

    dir_path: Path

    def path_string(self, path: str) -> str:
        return (self.dir_path / path).as_posix()

    def json_at(self, path: str) -> StorageFileSource:
        return StorageFileSource(path=self.path_string(path))

    def csv_at(
        self, path: str, mapping_keys: dict[str, str] | None = None, csv_config: CsvConfig | None = None
    ) -> CsvFileSource:
        return CsvFileSource(
            self.path_string(path), mapping_keys=mapping_keys or {}, csv_config=csv_config or CsvConfig()
        )

    def parquet_at(
        self, path: str, mapping_keys: dict[str, str] | None = None, config: ParquetConfig | None = None
    ) -> ParquetFileSource:
        return ParquetFileSource(
            path=self.path_string(path), mapping_keys=mapping_keys or {}, config=config or ParquetConfig()
        )

    def delta_at(
        self, path: str, mapping_keys: dict[str, str] | None = None, config: DeltaFileConfig | None = None
    ) -> DeltaFileSource:
        return DeltaFileSource(self.path_string(path), mapping_keys or {}, config=config or DeltaFileConfig())

    def directory(self, path: str) -> FileDirectory:
        return FileDirectory(self.dir_path / path)

    def repo_from_dir(self, dir: str, exclude: list[str] | None = None) -> AsRepoDefinition:
        return DirectoryRepo(Path(dir), exclude)


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
        path: str,
        mapping_keys: dict[str, str] | None = None,
        csv_config: CsvConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> CsvFileSource:
        return CsvFileSource(
            path,
            mapping_keys=mapping_keys or {},
            csv_config=csv_config or CsvConfig(),
            formatter=date_formatter or DateFormatter.iso_8601(),
        )

    @staticmethod
    def parquet_at(
        path: str, mapping_keys: dict[str, str] | None = None, config: ParquetConfig | None = None
    ) -> ParquetFileSource:
        return ParquetFileSource(path=path, mapping_keys=mapping_keys or {}, config=config or ParquetConfig())

    @staticmethod
    def delta_at(
        path: str, mapping_keys: dict[str, str] | None = None, config: DeltaFileConfig | None = None
    ) -> DeltaFileSource:
        return DeltaFileSource(path, mapping_keys or {}, config=config or DeltaFileConfig())

    @staticmethod
    def directory(path: str) -> FileDirectory:
        return FileDirectory(Path(path))

    @staticmethod
    def repo_from_dir(dir: str, exclude: list[str] | None = None) -> AsRepoDefinition:

        return DirectoryRepo(Path(dir), exclude)


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

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return self.file
