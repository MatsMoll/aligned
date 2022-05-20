from dataclasses import dataclass
from aioaws._types import S3ConfigProtocol
from aladdin.codable import Codable
from aladdin.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable


class AwsS3Config(Codable, S3ConfigProtocol):

    access_token_env: str
    secret_token_env: str
    bucket_env: str
    region_env: str

    @property
    def aws_access_key(self) -> str:
        import os
        return os.environ[self.access_token_env]
    
    @property
    def aws_secret_key(self) -> str:
        import os
        return os.environ[self.secret_token_env]

    @property
    def aws_s3_bucket(self) -> str:
        import os
        return os.environ[self.bucket_env]

    @property
    def aws_region(self) -> str:
        import os
        return os.environ[self.region_env]

    
    def file_at(self, path: str) -> "AwsS3DataSource":
        return AwsS3DataSource(
            config=self,
            path=path
        )


@dataclass
class AwsS3DataSource(BatchDataSource, ColumnFeatureMappable):

    config: AwsS3Config
    path: str
    mapping_keys: dict[str, str]

    type_name: str = "aws_s3"

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path}"