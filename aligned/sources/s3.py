from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

import polars as pl
from httpx import HTTPStatusError

from aligned.config_value import ConfigValue
from aligned.lazy_imports import pandas as pd
from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    ColumnFeatureMappable,
)
from aligned.exceptions import UnableToFindFileException
from aligned.local.job import FileDateJob, FileFullJob
from aligned.retrieval_job import RetrievalRequest, RetrievalJob
from aligned.s3.storage import AwsS3Storage
from aligned.schemas.codable import Codable
from aligned.sources.local import (
    CsvConfig,
    DataFileReference,
    ParquetConfig,
    PartitionedParquetFileSource,
    StorageFileReference,
    Directory,
    DeltaFileConfig,
    DateFormatter,
)
from aligned.storage import Storage

try:
    from aioaws.s3 import S3Config  # type: ignore
except ModuleNotFoundError:

    @dataclass
    class S3Config:  # type: ignore[no-redef]
        aws_access_key: str
        aws_secret_key: str
        aws_region: str
        aws_s3_bucket: str


@dataclass
class AwsS3Config(Codable):
    access_token_env: str
    secret_token_env: str
    bucket_env: str
    region_env: str

    @property
    def s3_config(self) -> S3Config:
        import os

        return S3Config(
            aws_access_key=os.environ[self.access_token_env],
            aws_secret_key=os.environ[self.secret_token_env],
            aws_region=os.environ[self.region_env],
            aws_s3_bucket=os.environ[self.bucket_env],
        )

    @property
    def url(self) -> str:
        import os

        region = os.environ[self.bucket_env]
        bucket = os.environ[self.bucket_env]
        return f"https://{region}.amazoneaws.com/{bucket}/"

    def json_at(
        self, path: str, mapping_keys: dict[str, str] | None = None
    ) -> "AwsS3DataSource":
        return AwsS3DataSource(config=self, path=path)

    def csv_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        csv_config: CsvConfig | None = None,
    ) -> "AwsS3CsvDataSource":
        return AwsS3CsvDataSource(
            config=self,
            path=path,
            mapping_keys=mapping_keys or {},
            csv_config=csv_config or CsvConfig(),
        )

    def parquet_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
    ) -> "AwsS3ParquetDataSource":
        return AwsS3ParquetDataSource(
            config=self,
            path=path,
            mapping_keys=mapping_keys or {},
            parquet_config=config or ParquetConfig(),
        )

    def sub_directory(self, path: str | ConfigValue) -> "AwsS3Directory":
        assert isinstance(path, str)
        return AwsS3Directory(config=self, path=path)

    def directory(self, path: str | ConfigValue) -> "AwsS3Directory":
        return self.sub_directory(path)

    def with_schema_version(
        self, sub_directory: str | ConfigValue | None = None
    ) -> Directory:
        raise NotImplementedError(type(self))

    def delta_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: DeltaFileConfig | None = None,
    ) -> CodableBatchDataSource:
        raise NotImplementedError(type(self))

    @property
    def storage(self) -> Storage:
        return AwsS3Storage(self)


@dataclass
class AwsS3Directory(Directory):
    config: AwsS3Config
    path: str

    def json_at(self, path: str) -> "AwsS3DataSource":
        return AwsS3DataSource(config=self.config, path=f"{self.path}/{path}")

    def csv_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        csv_config: CsvConfig | None = None,
    ) -> "AwsS3CsvDataSource":
        return AwsS3CsvDataSource(
            config=self.config,
            path=f"{self.path}/{path}",
            mapping_keys=mapping_keys or {},
            csv_config=csv_config or CsvConfig(),
        )

    def parquet_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> "AwsS3ParquetDataSource":
        return AwsS3ParquetDataSource(
            config=self.config,
            path=f"{self.path}/{path}",
            mapping_keys=mapping_keys or {},
            parquet_config=config or ParquetConfig(),
        )

    def partitioned_parquet_at(
        self,
        directory: str,
        partition_keys: list[str],
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> PartitionedParquetFileSource:
        raise NotImplementedError(type(self))

    def sub_directory(self, path: str | ConfigValue) -> "AwsS3Directory":
        return AwsS3Directory(config=self.config, path=f"{self.path}/{path}")

    def directory(self, path: str | ConfigValue) -> "AwsS3Directory":
        return self.sub_directory(path)

    def with_schema_version(
        self, sub_directory: str | ConfigValue | None = None
    ) -> Directory:
        raise NotImplementedError(type(self))

    def delta_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: DeltaFileConfig | None = None,
    ) -> CodableBatchDataSource:
        raise NotImplementedError(type(self))


@dataclass
class AwsS3DataSource(StorageFileReference, ColumnFeatureMappable):
    config: AwsS3Config
    path: str

    type_name: str = "aws_s3"

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path}"

    @property
    def storage(self) -> Storage:
        return self.config.storage

    @property
    def url(self) -> str:
        return f"{self.config.url}{self.path}"

    async def read(self) -> bytes:
        return await self.storage.read(self.path)

    async def write(self, content: bytes) -> None:
        return await self.storage.write(self.path, content)


@dataclass
class AwsS3CsvDataSource(
    CodableBatchDataSource, DataFileReference, ColumnFeatureMappable
):
    config: AwsS3Config
    path: str
    mapping_keys: dict[str, str]
    csv_config: CsvConfig

    type_name: str = "aws_s3_csv"

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path}"

    @property
    def storage(self) -> Storage:
        return self.config.storage

    @property
    def url(self) -> str:
        return f"{self.config.url}{self.path}"

    async def read_pandas(self) -> pd.DataFrame:
        try:
            data = await self.storage.read(self.path)
            buffer = BytesIO(data)
            return pd.read_csv(
                buffer,
                sep=self.csv_config.separator,
                compression=self.csv_config.compression,
            )
        except FileNotFoundError:
            raise UnableToFindFileException(self.path)
        except HTTPStatusError:
            raise UnableToFindFileException(self.path)

    async def write_pandas(self, df: pd.DataFrame) -> None:
        buffer = BytesIO()
        df.to_csv(
            buffer,
            sep=self.csv_config.separator,
            index=self.csv_config.should_write_index,
            compression=self.csv_config.compression,
        )
        buffer.seek(0)
        await self.storage.write(self.path, buffer.read())

    async def write_polars(self, df: pl.LazyFrame) -> None:
        buffer = BytesIO()
        df.collect().write_csv(
            buffer,
            separator=self.csv_config.separator,
        )
        buffer.seek(0)
        await self.storage.write(self.path, buffer.read())

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return FileFullJob(self, request=request, limit=limit)

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        return FileDateJob(self, request, start_date, end_date)


@dataclass
class AwsS3ParquetDataSource(
    CodableBatchDataSource, DataFileReference, ColumnFeatureMappable
):
    config: AwsS3Config
    path: str
    mapping_keys: dict[str, str]

    parquet_config: ParquetConfig
    type_name: str = "aws_s3_parquet"

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path}"

    @property
    def storage(self) -> Storage:
        return self.config.storage

    @property
    def url(self) -> str:
        return f"{self.config.url}{self.path}"

    async def read_pandas(self) -> pd.DataFrame:
        try:
            data = await self.storage.read(self.path)
            buffer = BytesIO(data)
            return pd.read_parquet(buffer)
        except FileNotFoundError:
            raise UnableToFindFileException(self.path)
        except HTTPStatusError:
            raise UnableToFindFileException(self.path)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        try:
            data = await self.storage.read(self.path)
            buffer = BytesIO(data)
            return pl.read_parquet(buffer).lazy()
        except FileNotFoundError:
            raise UnableToFindFileException(self.path)
        except HTTPStatusError:
            raise UnableToFindFileException(self.path)

    async def write_pandas(self, df: pd.DataFrame) -> None:
        buffer = BytesIO()
        df.to_parquet(
            buffer,
            compression=self.parquet_config.compression,
            engine=self.parquet_config.engine,
        )
        buffer.seek(0)
        await self.storage.write(self.path, buffer.read())

    async def write_polars(self, df: pl.LazyFrame) -> None:
        buffer = BytesIO()
        df.collect().write_parquet(buffer, compression=self.parquet_config.compression)
        buffer.seek(0)
        await self.storage.write(self.path, buffer.read())
