from dataclasses import dataclass
from io import BytesIO

import pandas as pd
from aioaws.s3 import S3Config
from httpx import HTTPStatusError

from aligned.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aligned.exceptions import UnableToFindFileException
from aligned.local.source import CsvConfig, DataFileReference, ParquetConfig, StorageFileReference
from aligned.s3.storage import AwsS3Storage
from aligned.schemas.codable import Codable
from aligned.storage import Storage


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
        return f'https://{region}.amazoneaws.com/{bucket}/'

    def file_at(self, path: str, mapping_keys: dict[str, str] | None = None) -> 'AwsS3DataSource':
        return AwsS3DataSource(config=self, path=path)

    def csv_at(
        self, path: str, mapping_keys: dict[str, str] | None = None, csv_config: CsvConfig | None = None
    ) -> 'AwsS3CsvDataSource':
        return AwsS3CsvDataSource(
            config=self, path=path, mapping_keys=mapping_keys or {}, csv_config=csv_config or CsvConfig()
        )

    def parquet_at(
        self, path: str, mapping_keys: dict[str, str] | None = None, config: ParquetConfig | None = None
    ) -> 'AwsS3ParquetDataSource':
        return AwsS3ParquetDataSource(
            config=self, path=path, mapping_keys=mapping_keys or {}, parquet_config=config or ParquetConfig()
        )

    @property
    def storage(self) -> Storage:
        return AwsS3Storage(self)


@dataclass
class AwsS3DataSource(StorageFileReference, ColumnFeatureMappable):

    config: AwsS3Config
    path: str

    type_name: str = 'aws_s3'

    def job_group_key(self) -> str:
        return f'{self.type_name}/{self.path}'

    @property
    def storage(self) -> Storage:
        return self.config.storage

    @property
    def url(self) -> str:
        return f'{self.config.url}{self.path}'

    async def read(self) -> bytes:
        return await self.storage.read(self.path)

    async def write(self, content: bytes) -> None:
        return await self.storage.write(self.path, content)


@dataclass
class AwsS3CsvDataSource(BatchDataSource, DataFileReference, ColumnFeatureMappable):

    config: AwsS3Config
    path: str
    mapping_keys: dict[str, str]
    csv_config: CsvConfig

    type_name: str = 'aws_s3_csv'

    def job_group_key(self) -> str:
        return f'{self.type_name}/{self.path}'

    @property
    def storage(self) -> Storage:
        return self.config.storage

    @property
    def url(self) -> str:
        return f'{self.config.url}{self.path}'

    async def read_pandas(self) -> pd.DataFrame:
        try:
            data = await self.storage.read(self.path)
            buffer = BytesIO(data)
            return pd.read_csv(buffer, sep=self.csv_config.seperator, compression=self.csv_config.compression)
        except FileNotFoundError:
            raise UnableToFindFileException()
        except HTTPStatusError:
            raise UnableToFindFileException()

    async def write_pandas(self, df: pd.DataFrame) -> None:
        buffer = BytesIO()
        df.to_csv(buffer, sep=self.csv_config.seperator, index=self.csv_config.should_write_index, compression=self.csv_config.compression)
        buffer.seek(0)
        await self.storage.write(self.path, buffer.read())


@dataclass
class AwsS3ParquetDataSource(BatchDataSource, DataFileReference, ColumnFeatureMappable):

    config: AwsS3Config
    path: str
    mapping_keys: dict[str, str]  # = field(default_factory=dict)

    parquet_config: ParquetConfig  # = field(default_factory=ParquetConfig)
    type_name: str = 'aws_s3_parquet'  # = field(default='aws_s3_parquet')

    def job_group_key(self) -> str:
        return f'{self.type_name}/{self.path}'

    @property
    def storage(self) -> Storage:
        return self.config.storage

    @property
    def url(self) -> str:
        return f'{self.config.url}{self.path}'

    async def read_pandas(self) -> pd.DataFrame:
        try:
            data = await self.storage.read(self.path)
            buffer = BytesIO(data)
            return pd.read_parquet(buffer)
        except FileNotFoundError:
            raise UnableToFindFileException()
        except HTTPStatusError:
            raise UnableToFindFileException()

    async def write_pandas(self, df: pd.DataFrame) -> None:
        buffer = BytesIO()
        df.to_parquet(buffer)
        buffer.seek(0)
        await self.storage.write(self.path, buffer.read())
