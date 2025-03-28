from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    ColumnFeatureMappable,
)
from aligned.exceptions import UnableToFindFileException
from aligned.feature_source import WritableFeatureSource
from aligned.local.job import FileDateJob, FileFactualJob, FileFullJob
from aligned.retrieval_job import RetrievalJob, RetrievalRequest
from aligned.schemas.date_formatter import DateFormatter
from aligned.schemas.feature import FeatureType, Feature
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.sources.local import (
    CsvConfig,
    DataFileReference,
    Deletable,
    DeltaFileConfig,
    ParquetConfig,
    StorageFileReference,
    Directory,
    data_file_freshness,
    upsert_on_column,
)
from aligned.storage import Storage
from httpx import HTTPStatusError
from aligned.lazy_imports import pandas as pd
from aligned.config_value import (
    ConfigValue,
    EnvironmentValue,
    NothingValue,
    PathResolver,
    PlaceholderValue,
)

if TYPE_CHECKING:
    from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


@dataclass
class AzurePath:
    container: str
    blob_path: str


def azure_container_blob(path: str) -> AzurePath:
    splits = path.split("/")
    return AzurePath(container=splits[0], blob_path="/".join(splits[1:]))


@dataclass
class AzureBlobConfig(Directory):
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
        return AzureBlobDirectory(self, PathResolver.from_value("")).json_at(path)

    def parquet_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> AzureBlobParquetDataSource:
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
        config: DeltaFileConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> AzureBlobDeltaDataSource:
        return AzureBlobDeltaDataSource(
            self,
            PathResolver.from_value(path),
            mapping_keys=mapping_keys or {},
            date_formatter=date_formatter or DateFormatter.unix_timestamp(),
        )

    def directory(self, path: str | ConfigValue) -> AzureBlobDirectory:
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
        return AzureBlobDirectory(
            self, PathResolver.from_value("")
        ).with_schema_version(sub_directory)

    @property
    def storage(self) -> BlobStorage:
        return BlobStorage(self)


class AzureConfigurable:
    "Something that contains an azure config"

    config: AzureBlobConfig


@dataclass
class AzureBlobDirectory(Directory):
    config: AzureBlobConfig
    components: PathResolver

    @classmethod
    def schema_placeholder(cls) -> ConfigValue:
        return PlaceholderValue("schema_version_placeholder")

    def json_at(self, path: str) -> StorageFileReference:
        return AzureBlobDataSource(self.config, self.components.append(path))

    def parquet_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> AzureBlobParquetDataSource:
        sub_path = self.components.append(path)
        return self.config.parquet_at(
            sub_path,  # type: ignore
            mapping_keys=mapping_keys,
            config=config,
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
        sub_path = self.components.append(directory)
        return self.config.partitioned_parquet_at(
            sub_path,  # type: ignore
            partition_keys,
            mapping_keys=mapping_keys,
            config=config,
            date_formatter=date_formatter,
        )

    def csv_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        csv_config: CsvConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> AzureBlobCsvDataSource:
        sub_path = self.components.append(path)
        return self.config.csv_at(
            sub_path,  # type: ignore
            mapping_keys=mapping_keys,
            date_formatter=date_formatter or DateFormatter.unix_timestamp(),
            csv_config=csv_config or CsvConfig(),
        )

    def delta_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: DeltaFileConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> AzureBlobDeltaDataSource:
        sub_path = self.components.append(path)
        return self.config.delta_at(
            sub_path.as_posix(),
            mapping_keys,
            config=config,
            date_formatter=date_formatter,  # type: ignore
        )

    def sub_directory(self, path: str | ConfigValue) -> AzureBlobDirectory:
        return AzureBlobDirectory(self.config, self.components.append(path))

    def directory(self, path: str | ConfigValue) -> AzureBlobDirectory:
        return AzureBlobDirectory(self.config, self.components.append(path))

    def with_schema_version(
        self, sub_directory: str | ConfigValue | None = None
    ) -> Directory:
        if sub_directory:
            return AzureBlobDirectory(
                self.config,
                self.components.append(sub_directory).append(
                    AzureBlobDirectory.schema_placeholder()
                ),
            )
        else:
            return AzureBlobDirectory(
                self.config,
                self.components.append(AzureBlobDirectory.schema_placeholder()),
            )


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
class AzureBlobDataSource(StorageFileReference, ColumnFeatureMappable):
    config: AzureBlobConfig
    path: PathResolver

    type_name: str = "azure_blob"

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path.as_posix()}"

    @property
    def storage(self) -> Storage:
        return self.config.storage

    async def read(self) -> bytes:
        return await self.storage.read(self.path.as_posix())

    async def write(self, content: bytes) -> None:
        return await self.storage.write(self.path.as_posix(), content)


@dataclass
class AzureBlobCsvDataSource(
    CodableBatchDataSource, DataFileReference, ColumnFeatureMappable, AzureConfigurable
):
    config: AzureBlobConfig
    path: PathResolver
    mapping_keys: dict[str, str] = field(default_factory=dict)
    csv_config: CsvConfig = field(default_factory=CsvConfig)
    date_formatter: DateFormatter = field(
        default_factory=lambda: DateFormatter.unix_timestamp()
    )

    type_name: str = "azure_blob_csv"

    @property
    def as_markdown(self) -> str:
        return f"""Type: *Azure Blob Csv File*

Path: *{self.path}*

{self.config.as_markdown}"""

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path.as_posix()}"

    def needed_configs(self) -> list[ConfigValue]:
        return self.config.needed_configs()

    @property
    def storage(self) -> Storage:
        return self.config.storage

    def with_view(self, view: CompiledFeatureView) -> AzureBlobCsvDataSource:
        schema_hash = view.schema_hash()
        return AzureBlobCsvDataSource(
            config=self.config,
            path=self.path.replace(
                AzureBlobDirectory.schema_placeholder(), schema_hash.hex()
            ),
            mapping_keys=self.mapping_keys,
            csv_config=self.csv_config,
            date_formatter=self.date_formatter,
        )

    async def schema(self) -> dict[str, FeatureType]:
        try:
            schema = (await self.to_lazy_polars()).schema
            return {
                name: FeatureType.from_polars(pl_type)
                for name, pl_type in schema.items()
            }

        except FileNotFoundError as error:
            raise UnableToFindFileException(self.path.as_posix()) from error
        except HTTPStatusError as error:
            raise UnableToFindFileException(self.path.as_posix()) from error

    async def to_lazy_polars(self) -> pl.LazyFrame:
        url = f"az://{self.path}"
        return pl.scan_csv(
            url,
            separator=self.csv_config.separator,
            storage_options=self.config.read_creds(),
        )

    async def to_pandas(self) -> pd.DataFrame:
        path = self.path.as_posix()
        try:
            data = await self.storage.read(path)
            buffer = BytesIO(data)
            return pd.read_csv(
                buffer,
                sep=self.csv_config.separator,
                compression=self.csv_config.compression,
            )
        except FileNotFoundError as error:
            raise UnableToFindFileException(path) from error
        except HTTPStatusError as error:
            raise UnableToFindFileException(path) from error

    async def write_pandas(self, df: pd.DataFrame) -> None:
        url = f"az://{self.path}"
        df.to_csv(
            url,
            sep=self.csv_config.separator,
            compression=self.csv_config.compression,
            storage_options=self.config.read_creds(),
        )

    async def write_polars(self, df: pl.LazyFrame) -> None:
        await self.write_pandas(df.collect().to_pandas())

    async def write(self, job: RetrievalJob, requests: list[RetrievalRequest]) -> None:
        if len(requests) != 1:
            raise ValueError(f"Only support writing on request, got {len(requests)}.")

        features = requests[0].all_returned_columns
        df = await job.to_lazy_polars()
        await self.write_polars(df.select(features))

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls,
        facts: RetrievalJob,
        requests: list[tuple[AzureBlobCsvDataSource, RetrievalRequest]],
    ) -> RetrievalJob:
        source = requests[0][0]
        if not isinstance(source, cls):
            raise ValueError(f"Only {cls} is supported, received: {source}")

        # Group based on config
        return FileFactualJob(
            source=source,
            requests=[request for _, request in requests],
            facts=facts,
            date_formatter=source.date_formatter,
        )

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        return FileFactualJob(
            self, [request], facts, date_formatter=self.date_formatter
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return FileFullJob(self, request, limit, date_formatter=self.date_formatter)

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        return FileDateJob(
            source=self,
            request=request,
            start_date=start_date,
            end_date=end_date,
            date_formatter=self.date_formatter,
        )


@dataclass
class AzureBlobPartitionedParquetDataSource(
    CodableBatchDataSource,
    DataFileReference,
    ColumnFeatureMappable,
    Deletable,
    WritableFeatureSource,
    AzureConfigurable,
):
    config: AzureBlobConfig
    directory: PathResolver
    partition_keys: list[str]
    mapping_keys: dict[str, str] = field(default_factory=dict)
    parquet_config: ParquetConfig = field(default_factory=ParquetConfig)
    date_formatter: DateFormatter = field(default_factory=lambda: DateFormatter.noop())
    type_name: str = "azure_blob_partitiond_parquet"

    @property
    def as_markdown(self) -> str:
        return f"""Type: *Azure Blob Partitioned Parquet File*

Directory: *{self.directory}*
Partition Keys: *{self.partition_keys}*

{self.config.as_markdown}"""

    def needed_configs(self) -> list[ConfigValue]:
        return self.config.needed_configs()

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.directory.as_posix()}"

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    def with_view(
        self, view: CompiledFeatureView
    ) -> AzureBlobPartitionedParquetDataSource:
        schema_hash = view.schema_hash()
        return AzureBlobPartitionedParquetDataSource(
            config=self.config,
            directory=self.directory.replace(
                AzureBlobDirectory.schema_placeholder(), schema_hash.hex()
            ),
            partition_keys=self.partition_keys,
            mapping_keys=self.mapping_keys,
            parquet_config=self.parquet_config,
            date_formatter=self.date_formatter,
        )

    @property
    def storage(self) -> Storage:
        return self.config.storage

    async def schema(self) -> dict[str, FeatureType]:
        try:
            schema = (await self.to_lazy_polars()).schema
            return {
                name: FeatureType.from_polars(pl_type)
                for name, pl_type in schema.items()
            }

        except FileNotFoundError as error:
            raise UnableToFindFileException(self.directory.as_posix()) from error
        except HTTPStatusError as error:
            raise UnableToFindFileException(self.directory.as_posix()) from error

    async def read_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        try:
            url = f"az://{self.directory}/**/*.parquet"
            creds = self.config.read_creds()
            return pl.scan_parquet(url, storage_options=creds, hive_partitioning=True)
        except FileNotFoundError as error:
            raise UnableToFindFileException(self.directory.as_posix()) from error
        except HTTPStatusError as error:
            raise UnableToFindFileException(self.directory.as_posix()) from error
        except pl.ComputeError as error:
            raise UnableToFindFileException(self.directory.as_posix()) from error

    async def write_pandas(self, df: pd.DataFrame) -> None:
        await self.write_polars(pl.from_pandas(df).lazy())

    async def write_polars(self, df: pl.LazyFrame) -> None:
        from adlfs import AzureBlobFileSystem
        from pyarrow.parquet import write_to_dataset

        fs = AzureBlobFileSystem(**self.config.read_creds())  # type: ignore

        pyarrow_options = {
            "partition_cols": self.partition_keys,
            "filesystem": fs,
            "compression": "zstd",
        }

        write_to_dataset(
            table=df.collect().to_arrow(),
            root_path=self.directory,
            **(pyarrow_options or {}),
        )

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls,
        facts: RetrievalJob,
        requests: list[tuple[AzureBlobParquetDataSource, RetrievalRequest]],
    ) -> RetrievalJob:
        source = requests[0][0]
        if not isinstance(source, cls):
            raise ValueError(f"Only {cls} is supported, received: {source}")

        # Group based on config
        return FileFactualJob(
            source=source,
            requests=[request for _, request in requests],
            facts=facts,
            date_formatter=source.date_formatter,
        )

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        return FileFactualJob(
            self, [request], facts, date_formatter=self.date_formatter
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return FileFullJob(self, request, limit, date_formatter=self.date_formatter)

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        return FileDateJob(
            source=self,
            request=request,
            start_date=start_date,
            end_date=end_date,
            date_formatter=self.date_formatter,
        )

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        features = request.all_returned_columns
        df = await job.select(features).to_lazy_polars()
        await self.write_polars(df)

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        from adlfs import AzureBlobFileSystem

        fs = AzureBlobFileSystem(**self.config.read_creds())  # type: ignore

        def delete_directory_recursively(directory_path: str) -> None:
            paths = fs.find(directory_path)

            for path in paths:
                if fs.info(path)["type"] == "directory":
                    delete_directory_recursively(path)
                else:
                    fs.rm(path)

            fs.rmdir(directory_path)

        upsert_on = sorted(request.entity_names)
        returend_columns = request.all_returned_columns

        df = await job.select(returend_columns).to_polars()
        if df.is_empty():
            return

        unique_partitions = df.select(self.partition_keys).unique()

        filters: list[pl.Expr] = []
        for row in unique_partitions.iter_rows(named=True):
            current: pl.Expr | None = None

            for key, value in row.items():
                if current is not None:
                    current = current & (pl.col(key) == value)
                else:
                    current = pl.col(key) == value

            if current is not None:
                filters.append(current)

        try:
            existing_df = (await self.to_lazy_polars()).filter(*filters)
            write_df = upsert_on_column(upsert_on, df.lazy(), existing_df).collect()
        except (UnableToFindFileException, pl.ComputeError):
            write_df = df.lazy()

        for row in unique_partitions.iter_rows(named=True):
            dir = Path(self.directory.as_posix())
            for partition_key in self.partition_keys:
                dir = dir / f"{partition_key}={row[partition_key]}"

            if fs.exists(dir.as_posix()):
                delete_directory_recursively(dir.as_posix())

        await self.write_polars(write_df.select(returend_columns).lazy())

    async def delete(self) -> None:
        from adlfs import AzureBlobFileSystem

        fs = AzureBlobFileSystem(**self.config.read_creds())  # type: ignore

        def delete_directory_recursively(directory_path: str) -> None:
            paths = fs.find(directory_path)

            for path in paths:
                if fs.info(path)["type"] == "directory":
                    delete_directory_recursively(path)
                else:
                    fs.rm(path)

            fs.rmdir(directory_path)

        delete_directory_recursively(self.directory.as_posix())

    async def overwrite(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        await self.delete()
        await self.insert(job, request)


@dataclass
class AzureBlobParquetDataSource(
    CodableBatchDataSource, DataFileReference, ColumnFeatureMappable, AzureConfigurable
):
    config: AzureBlobConfig
    path: PathResolver
    mapping_keys: dict[str, str] = field(default_factory=dict)
    parquet_config: ParquetConfig = field(default_factory=ParquetConfig)
    date_formatter: DateFormatter = field(default_factory=lambda: DateFormatter.noop())
    type_name: str = "azure_blob_parquet"

    @property
    def as_markdown(self) -> str:
        return f"""Type: *Azure Blob Parquet File*

Path: *{self.path}*

{self.config.as_markdown}"""

    def needed_configs(self) -> list[ConfigValue]:
        return self.config.needed_configs()

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path.as_posix()}"

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    def with_view(self, view: CompiledFeatureView) -> AzureBlobParquetDataSource:
        return AzureBlobParquetDataSource(
            config=self.config,
            path=self.path.replace(
                AzureBlobDirectory.schema_placeholder(), view.schema_hash().hex()
            ),
            mapping_keys=self.mapping_keys,
            parquet_config=self.parquet_config,
            date_formatter=self.date_formatter,
        )

    @property
    def storage(self) -> Storage:
        return self.config.storage

    async def schema(self) -> dict[str, FeatureType]:
        try:
            schema = (await self.to_lazy_polars()).schema
            return {
                name: FeatureType.from_polars(pl_type)
                for name, pl_type in schema.items()
            }

        except FileNotFoundError as error:
            raise UnableToFindFileException(self.path.as_posix()) from error
        except HTTPStatusError as error:
            raise UnableToFindFileException(self.path.as_posix()) from error

    async def read_pandas(self) -> pd.DataFrame:
        path = self.path.as_posix()
        try:
            data = await self.storage.read(path)
            buffer = BytesIO(data)
            return pd.read_parquet(buffer)
        except FileNotFoundError as error:
            raise UnableToFindFileException(path) from error
        except HTTPStatusError as error:
            raise UnableToFindFileException(path) from error

    async def to_lazy_polars(self) -> pl.LazyFrame:
        url = f"az://{self.path}"
        try:
            creds = self.config.read_creds()
            return pl.scan_parquet(url, storage_options=creds)
        except FileNotFoundError as error:
            raise UnableToFindFileException(url) from error
        except HTTPStatusError as error:
            raise UnableToFindFileException(url) from error
        except pl.ComputeError as error:
            raise UnableToFindFileException(url) from error

    async def write_pandas(self, df: pd.DataFrame) -> None:
        buffer = BytesIO()
        df.to_parquet(
            buffer,
            compression=self.parquet_config.compression,
            engine=self.parquet_config.engine,
        )
        buffer.seek(0)
        await self.storage.write(self.path.as_posix(), buffer.read())

    async def write_polars(self, df: pl.LazyFrame) -> None:
        url = f"az://{self.path}"
        creds = self.config.read_creds()
        df.collect().to_pandas().to_parquet(url, storage_options=creds)

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls,
        facts: RetrievalJob,
        requests: list[tuple[AzureBlobParquetDataSource, RetrievalRequest]],
    ) -> RetrievalJob:
        source = requests[0][0]
        if not isinstance(source, cls):
            raise ValueError(f"Only {cls} is supported, received: {source}")

        # Group based on config
        return FileFactualJob(
            source=source,
            requests=[request for _, request in requests],
            facts=facts,
            date_formatter=source.date_formatter,
        )

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        return FileFactualJob(
            self, [request], facts, date_formatter=self.date_formatter
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return FileFullJob(self, request, limit, date_formatter=self.date_formatter)

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        return FileDateJob(
            source=self,
            request=request,
            start_date=start_date,
            end_date=end_date,
            date_formatter=self.date_formatter,
        )


@dataclass
class AzureBlobDeltaDataSource(
    CodableBatchDataSource,
    DataFileReference,
    ColumnFeatureMappable,
    WritableFeatureSource,
    Deletable,
    AzureConfigurable,
):
    config: AzureBlobConfig
    path: PathResolver
    mapping_keys: dict[str, str] = field(default_factory=dict)
    date_formatter: DateFormatter = field(
        default_factory=lambda: DateFormatter.unix_timestamp("ms")
    )
    type_name: str = "azure_blob_delta"

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path.as_posix()}"

    @property
    def as_markdown(self) -> str:
        return f"""Type: Azure Blob Delta File

Path: *{self.path}*

{self.config.as_markdown}"""

    def needed_configs(self) -> list[ConfigValue]:
        return self.config.needed_configs()

    @property
    def storage(self) -> Storage:
        return self.config.storage

    async def read_pandas(self) -> pd.DataFrame:
        return (await self.to_polars()).to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        url = f"az://{self.path.as_posix()}"
        try:
            creds = self.config.read_creds()
            return pl.scan_delta(url, storage_options=creds)
        except FileNotFoundError as error:
            raise UnableToFindFileException(url) from error
        except HTTPStatusError as error:
            raise UnableToFindFileException(url) from error

    async def freshness(self, feature: Feature) -> datetime | None:
        try:
            return await data_file_freshness(self, feature.name)
        except Exception as error:
            logger.info(
                f"Failed to get freshness for {self.path}. {error} - returning None."
            )
            return None

    async def schema(self) -> dict[str, FeatureType]:
        try:
            schema = (await self.to_lazy_polars()).schema
            return {
                name: FeatureType.from_polars(pl_type)
                for name, pl_type in schema.items()
            }

        except FileNotFoundError as error:
            raise UnableToFindFileException(self.path.as_posix()) from error
        except HTTPStatusError as error:
            raise UnableToFindFileException(self.path.as_posix()) from error

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls,
        facts: RetrievalJob,
        requests: list[tuple[AzureBlobDeltaDataSource, RetrievalRequest]],
    ) -> RetrievalJob:
        source = requests[0][0]
        if not isinstance(source, cls):
            raise ValueError(f"Only {cls} is supported, received: {source}")

        # Group based on config
        return FileFactualJob(
            source=source,
            requests=[request for _, request in requests],
            facts=facts,
            date_formatter=source.date_formatter,
        )

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        return FileFactualJob(
            self, [request], facts, date_formatter=self.date_formatter
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return FileFullJob(self, request, limit, date_formatter=self.date_formatter)

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        return FileDateJob(
            source=self,
            request=request,
            start_date=start_date,
            end_date=end_date,
            date_formatter=self.date_formatter,
        )

    async def write_pandas(self, df: pd.DataFrame) -> None:
        await self.write_polars(pl.from_pandas(df).lazy())

    async def write_polars(self, df: pl.LazyFrame) -> None:
        url = f"az://{self.path}"
        creds = self.config.read_creds()

        df.collect().write_delta(
            url,
            storage_options=creds,
            mode="append",
        )

    def df_to_deltalake_compatible(
        self, df: pl.DataFrame, request: RetrievalRequest
    ) -> tuple[pl.DataFrame, dict]:
        import pyarrow as pa
        from aligned.schemas.constraints import Optional

        def pa_dtype(dtype: FeatureType) -> pa.DataType:
            pa_types = {
                "int8": pa.int8(),
                "int16": pa.int16(),
                "int32": pa.int32(),
                "int64": pa.int64(),
                "float": pa.float64(),
                "double": pa.float64(),
                "string": pa.large_string(),
                "date": pa.date64(),
                "embedding": pa.large_list(pa.float32()),
                "datetime": pa.float64(),
                "list": pa.large_list(pa.int32()),
                "array": pa.large_list(pa.int32()),
                "bool": pa.bool_(),
            }

            if dtype.name in pa_types:
                return pa_types[dtype.name]

            if dtype.is_datetime:
                return pa.float64()

            if dtype.is_array:
                array_sub_dtype = dtype.array_subtype()
                if array_sub_dtype:
                    return pa.large_list(pa_dtype(array_sub_dtype))

                return pa.large_list(pa.string())

            raise ValueError(f"Unsupported dtype: {dtype}")

        def pa_field(feature: Feature) -> pa.Field:
            is_nullable = Optional() in (feature.constraints or set())

            pa_type = pa_dtype(feature.dtype)
            return pa.field(feature.name, pa_type, nullable=is_nullable)

        dtypes = dict(zip(df.columns, df.dtypes, strict=False))
        schemas = {}

        features = request.all_features.union(request.entities)
        if request.event_timestamp:
            features.add(request.event_timestamp.as_feature())

        for feature in features:
            schemas[feature.name] = pa_field(feature)

            if dtypes[feature.name] == pl.Null:
                df = df.with_columns(
                    pl.col(feature.name).cast(feature.dtype.polars_type)
                )
            elif feature.dtype.is_datetime:
                df = df.with_columns(self.date_formatter.encode_polars(feature.name))
            else:
                df = df.with_columns(
                    pl.col(feature.name).cast(feature.dtype.polars_type)
                )

        return df, schemas

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        import pyarrow as pa

        df = await job.to_polars()
        url = f"az://{self.path.as_posix()}"

        df, schemas = self.df_to_deltalake_compatible(df, request)

        orderd_schema = OrderedDict(sorted(schemas.items()))
        schema = list(orderd_schema.values())
        df.select(list(orderd_schema.keys())).write_delta(
            url,
            storage_options=self.config.read_creds(),
            mode="append",
            delta_write_options={"schema": pa.schema(schema)},
        )

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        import pyarrow as pa
        from deltalake.exceptions import TableNotFoundError

        df = await job.to_polars()

        url = f"az://{self.path.as_posix()}"
        merge_on = request.entity_names

        df, schemas = self.df_to_deltalake_compatible(df, request)

        orderd_schema = OrderedDict(sorted(schemas.items()))
        schema = list(orderd_schema.values())
        predicate = " AND ".join([f"s.{key} = t.{key}" for key in merge_on])

        try:
            from deltalake import DeltaTable

            table = DeltaTable(url, storage_options=self.config.read_creds())
            pa_df = (
                df.select(list(orderd_schema.keys())).to_arrow().cast(pa.schema(schema))
            )

            (
                table.merge(
                    pa_df,
                    predicate=predicate,
                    source_alias="s",
                    target_alias="t",
                )
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute()
            )

        except TableNotFoundError:
            df.write_delta(
                url,
                mode="append",
                storage_options=self.config.read_creds(),
                delta_write_options={"schema": pa.schema(schema)},
            )

    async def delete(self) -> None:
        from deltalake import DeltaTable

        url = f"az://{self.path.as_posix()}"
        table = DeltaTable(url, storage_options=self.config.read_creds())
        table.delete()

    async def overwrite(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        await self.delete()
        await self.insert(job, request)
