from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aligned.schemas.codable import Codable
from aligned.schemas.date_formatter import DateFormatter
from aligned.config_value import (
    ConfigValue,
    EnvironmentValue,
    NothingValue,
    PathResolver,
)
from aligned.storage import Storage
from io import BytesIO

if TYPE_CHECKING:
    from aligned.sources.local import DeltaConfig, DeltaFileSource
    from azure.storage.blob import BlobServiceClient
    from aligned.sources.azure_blob_storage import (
        AzureBlobParquetDataSource,
        AzureBlobPartitionedParquetDataSource,
        AzureBlobCsvDataSource,
        AzureBlobDirectory,
    )
    from aligned.sources.local import (
        CsvConfig,
        ParquetConfig,
        StorageFileReference,
        Directory,
    )


@dataclass
class AzurePath:
    container: str
    blob_path: str


def azure_container_blob(path: str) -> AzurePath:
    splits = path.split("/")
    return AzurePath(container=splits[0], blob_path="/".join(splits[1:]))


@dataclass
class BlobStorage(Storage):
    config: AzureBlobConfig

    async def read(self, path: str) -> bytes:
        azure_path = azure_container_blob(path)
        client = self.config.client()
        container = client.get_blob_client(azure_path.container, azure_path.blob_path)

        with BytesIO() as byte_stream:
            container.download_blob().download_to_stream(byte_stream)
            byte_stream.seek(0)
            return byte_stream.read()

    async def write(self, path: str, content: bytes | bytearray) -> None:
        azure_path = azure_container_blob(path)
        client = self.config.client()
        container = client.get_blob_client(azure_path.container, azure_path.blob_path)
        container.upload_blob(bytes(content), overwrite=True)


@dataclass
class AzureBlobConfig(Codable):
    account_id: ConfigValue
    tenant_id: ConfigValue
    client_id: ConfigValue
    client_secret: ConfigValue
    account_name: ConfigValue

    @property
    def as_markdown(self) -> str:
        return f"""Type: **Azure Blob Config**

You can choose between two ways of authenticating with Azure Blob Storage.

1. Using Account Name and Account Key

- Account Name: `{self.account_name}`
- Account Id: `{self.account_id}`

2. Using Tenant Id, Client Id and Client Secret

- Tenant Id: `{self.tenant_id}`
- Client Id: `{self.client_id}`
- Client Secret: `{self.client_secret}`
"""

    @staticmethod
    def from_account_id(
        account_id: str | ConfigValue, account_name: str | ConfigValue
    ) -> AzureBlobConfig:
        return AzureBlobConfig(
            account_id=ConfigValue.from_value(account_id),
            account_name=ConfigValue.from_value(account_name),
            tenant_id=NothingValue(),
            client_id=NothingValue(),
            client_secret=NothingValue(),
        )

    @staticmethod
    def from_tenant(
        tenant_id: str | ConfigValue,
        account_name: str | ConfigValue,
        client_id: str | ConfigValue,
        client_secret: str | ConfigValue,
    ) -> AzureBlobConfig:
        return AzureBlobConfig(
            account_id=NothingValue(),
            account_name=ConfigValue.from_value(account_name),
            tenant_id=ConfigValue.from_value(tenant_id),
            client_id=ConfigValue.from_value(client_id),
            client_secret=ConfigValue.from_value(client_secret),
        )

    def needed_configs(self) -> list[ConfigValue]:
        potential = [
            self.account_id,
            self.tenant_id,
            self.client_id,
            self.client_secret,
            self.account_name,
        ]
        return [val for val in potential if isinstance(val, EnvironmentValue)]

    def json_at(self, path: str) -> StorageFileReference:
        from aligned.sources.azure_blob_storage import AzureBlobDirectory

        return AzureBlobDirectory(self, PathResolver.from_value("")).json_at(path)

    def parquet_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> AzureBlobParquetDataSource:
        from aligned.sources.azure_blob_storage import AzureBlobParquetDataSource
        from aligned.sources.local import ParquetConfig

        return AzureBlobParquetDataSource(
            self,
            PathResolver.from_value(path),
            mapping_keys=mapping_keys or {},
            parquet_config=config or ParquetConfig(),
            date_formatter=date_formatter or DateFormatter.noop(),
        )

    def partitioned_parquet_at(
        self,
        directory: str,
        partition_keys: list[str],
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> AzureBlobPartitionedParquetDataSource:
        from aligned.sources.azure_blob_storage import (
            AzureBlobPartitionedParquetDataSource,
        )
        from aligned.sources.local import ParquetConfig

        return AzureBlobPartitionedParquetDataSource(
            self,
            PathResolver.from_value(directory),
            partition_keys,
            mapping_keys=mapping_keys or {},
            parquet_config=config or ParquetConfig(),
            date_formatter=date_formatter or DateFormatter.noop(),
        )

    def csv_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        csv_config: CsvConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> AzureBlobCsvDataSource:
        from aligned.sources.azure_blob_storage import AzureBlobCsvDataSource
        from aligned.sources.local import CsvConfig

        return AzureBlobCsvDataSource(
            self,
            PathResolver.from_value(path),
            mapping_keys=mapping_keys or {},
            date_formatter=date_formatter or DateFormatter.unix_timestamp(),
            csv_config=csv_config or CsvConfig(),
        )

    def delta_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: DeltaConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> DeltaFileSource:
        from aligned.sources.local import DeltaConfig

        return DeltaFileSource(
            path=PathResolver.from_value(path),
            config=config or DeltaConfig(),
            mapping_keys=mapping_keys or {},
            date_formatter=date_formatter or DateFormatter.unix_timestamp(),
            azure_config=self,
        )

    def directory(self, path: str | ConfigValue) -> AzureBlobDirectory:
        from aligned.sources.azure_blob_storage import AzureBlobDirectory

        return AzureBlobDirectory(self, PathResolver.from_value(path))

    def sub_directory(self, path: str | ConfigValue) -> Directory:
        return self.directory(path)

    def client(self) -> BlobServiceClient:
        from azure.storage.blob import BlobServiceClient

        creds = self.read_creds()
        account_name = creds["account_name"]
        account_url = f"https://{account_name}.blob.core.windows.net/"

        if "account_key" in creds:
            return BlobServiceClient(account_url=account_url, credential=creds)
        else:
            from azure.identity import ClientSecretCredential

            creds = ClientSecretCredential(
                tenant_id=creds["tenant_id"],
                client_id=creds["client_id"],
                client_secret=creds["client_secret"],
            )

            return BlobServiceClient(account_url=account_url, credential=creds)

    def read_creds(self) -> dict[str, str]:
        account_name = self.account_name.read()
        try:
            return {"account_name": account_name, "account_key": self.account_id.read()}
        except ValueError:
            return {
                "account_name": account_name,
                "tenant_id": self.tenant_id.read(),
                "client_id": self.client_id.read(),
                "client_secret": self.client_secret.read(),
            }

    def with_schema_version(
        self, sub_directory: str | ConfigValue | None = None
    ) -> Directory:
        from aligned.sources.azure_blob_storage import AzureBlobDirectory

        return AzureBlobDirectory(
            self, PathResolver.from_value("")
        ).with_schema_version(sub_directory)

    @property
    def storage(self) -> BlobStorage:
        return BlobStorage(self)
