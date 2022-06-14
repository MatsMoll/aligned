from dataclasses import dataclass

from aioaws.s3 import S3Config

from aladdin.codable import Codable
from aladdin.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aladdin.local.source import FileReference
from aladdin.s3.storage import AwsS3Storage
from aladdin.storage import Storage


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

    def file_at(self, path: str, mapping_keys: dict[str, str] | None = None) -> 'AwsS3DataSource':
        return AwsS3DataSource(config=self, path=path, mapping_keys=mapping_keys or {})

    @property
    def storage(self) -> Storage:
        return AwsS3Storage(self)


@dataclass
class AwsS3DataSource(BatchDataSource, ColumnFeatureMappable, FileReference):

    config: AwsS3Config
    path: str
    mapping_keys: dict[str, str]

    type_name: str = 'aws_s3'

    def job_group_key(self) -> str:
        return f'{self.type_name}/{self.path}'

    @property
    def storage(self) -> Storage:
        return self.config.storage

    async def read(self) -> bytes:
        return await self.storage.read(self.path)

    async def write(self, content: bytes) -> None:
        return await self.storage.write(self.path, content)
