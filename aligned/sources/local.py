from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

from aligned.config_value import ConfigValue, PathResolver, PlaceholderValue
from aligned.lazy_imports import pandas as pd
import polars as pl
from httpx import HTTPStatusError

from aligned.data_file import DataFileReference, upsert_on_column
from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    ColumnFeatureMappable,
)
from aligned.exceptions import UnableToFindFileException
from aligned.local.job import FileDateJob, FileFactualJob, FileFullJob
from aligned.request.retrieval_request import RetrievalRequest
from aligned.retrieval_job import RetrievalJob
from aligned.s3.storage import FileStorage, HttpStorage
from aligned.schemas.codable import Codable
from aligned.schemas.feature import FeatureType, Feature
from aligned.storage import Storage
from aligned.feature_source import WritableFeatureSource
from aligned.schemas.date_formatter import DateFormatter
from aligned.sources.azure_blob_config import AzureBlobConfig

if TYPE_CHECKING:
    from datetime import datetime
    from aligned.schemas.transformation import Expression
    from aligned.schemas.repo_definition import RepoDefinition
    from aligned.schemas.feature_view import CompiledFeatureView
    from aligned.feature_store import ContractStore


logger = logging.getLogger(__name__)


class Deletable:
    async def delete(self, predicate: Expression | None = None) -> None:
        raise NotImplementedError(type(self))


class AsRepoDefinition:
    async def as_repo_definition(self) -> RepoDefinition:
        raise NotImplementedError()

    async def as_contract_store(self) -> ContractStore:
        from aligned.feature_store import ContractStore

        return ContractStore.from_definition(await self.as_repo_definition())

    async def feature_store(self) -> ContractStore:
        return await self.as_contract_store()


class StorageFileReference(AsRepoDefinition):
    """
    A reference to a file that can be loaded as bytes.

    The bytes can contain anything, potentially a FeatureStore definition
    """

    def posix_path(self) -> str:
        raise NotImplementedError(type(self))

    async def read(self) -> bytes:
        raise NotImplementedError(type(self))

    async def write(self, content: bytes) -> None:
        raise NotImplementedError(type(self))

    async def as_repo_definition(self) -> RepoDefinition:
        from aligned.schemas.repo_definition import RepoDefinition

        file = await self.read()
        return RepoDefinition.from_json(file)


@dataclass
class AlignedCloudSource(AsRepoDefinition):
    host: str

    async def as_repo_definition(self) -> RepoDefinition:
        from aligned.schemas.repo_definition import RepoDefinition
        from httpx import AsyncClient

        url = f"{self.host}/api/definitions"

        client = AsyncClient()
        res = await client.get(url)
        res.raise_for_status()
        return RepoDefinition.from_json(res.content)


async def data_file_freshness(
    reference: DataFileReference,
    column_name: str,
    formatter: DateFormatter | None = None,
) -> datetime | None:
    try:
        formatter = formatter or DateFormatter.unix_timestamp()
        file = await reference.to_lazy_polars()
        if isinstance(reference, ColumnFeatureMappable):
            source_column = reference.feature_identifier_for([column_name])[0]
        else:
            source_column = column_name

        return (
            file.select(formatter.decode_polars(source_column))
            .max()
            .collect()[0, source_column]
        )
    except UnableToFindFileException:
        return None


def fill_missing_in_request(
    request: RetrievalRequest, df: pl.LazyFrame, feature_rename: dict[str, str]
) -> pl.LazyFrame:
    existing_columns = df.collect_schema()

    missing_features = [
        feature
        for feature in request.features
        if feature_rename.get(feature.name, feature.name) not in existing_columns
    ]

    if missing_features:
        return fill_with_default(missing_features, df, feature_rename)
    else:
        return df


def fill_with_default(
    features: list[Feature], df: pl.LazyFrame, feature_rename: dict[str, str]
) -> pl.LazyFrame:
    default_values = [
        (
            feature_rename.get(feature.name, feature.name),
            feature.default_value.python_value,
        )
        for feature in features
        if feature.default_value is not None
    ]

    if not default_values:
        return df

    return df.with_columns(
        [pl.lit(value).alias(feature_name) for feature_name, value in default_values]
    )


def create_parent_dir(path: str) -> None:
    parents = []

    file_path = Path(path)
    parent = file_path.parent

    while not parent.exists():
        parents.append(parent)
        parent = parent.parent

    for parent in reversed(parents):
        parent.mkdir(exist_ok=True)


def do_dir_exist(path: str) -> bool:
    return Path(path).is_dir()


def do_file_exist(path: str) -> bool:
    return Path(path).is_file()


def delete_path(path: str) -> None:
    path_obj = Path(path)

    if not path_obj.exists():
        return

    if path_obj.is_dir():
        shutil.rmtree(path)
    else:
        Path(path).unlink()


@dataclass
class CsvConfig(Codable):
    """
    A config for how a CSV file should be loaded
    """

    separator: str = field(default=",")
    compression: Literal["infer", "gzip", "bz2", "zip", "xz", "zstd"] = field(
        default="infer"
    )
    should_write_index: bool = field(default=False)


@dataclass
class CsvFileSource(
    CodableBatchDataSource,
    ColumnFeatureMappable,
    DataFileReference,
    WritableFeatureSource,
    Deletable,
):
    """
    A source pointing to a CSV file
    """

    path: PathResolver
    mapping_keys: dict[str, str] = field(default_factory=dict)
    csv_config: CsvConfig = field(default_factory=CsvConfig)
    formatter: DateFormatter = field(default_factory=DateFormatter.iso_8601)
    expected_schema: dict[str, FeatureType] | None = field(default=None)

    type_name: str = "csv"

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path.as_posix()}"

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    def with_view(self, view: CompiledFeatureView) -> CsvFileSource:
        schema_hash = view.schema_hash()
        return CsvFileSource(
            path=self.path.replace(
                FileDirectory.schema_placeholder(), schema_hash.hex()
            ),
            mapping_keys=self.mapping_keys,
            csv_config=self.csv_config,
            formatter=self.formatter,
            expected_schema=self.expected_schema,
        )

    @property
    def as_markdown(self) -> str:
        return f"""### CSV File

*Renames*: {self.mapping_keys}

*File*: {self.path.as_posix()}

*CSV Config*: {self.csv_config}

*Datetime Formatter*: {self.formatter}

[Go to file]({self.path.as_posix()})
"""  # noqa

    async def delete(self, predicate: Expression | None = None) -> None:
        if not predicate:
            delete_path(self.path.as_posix())
        else:
            polars_exp = predicate.to_polars()
            assert polars_exp is not None
            df = await self.to_lazy_polars()
            filtered_df = df.filter(polars_exp.not_())
            await self.write_polars(filtered_df)

    async def read_pandas(self) -> pd.DataFrame:
        path = self.path.as_posix()
        try:
            return pd.read_csv(
                path,
                sep=self.csv_config.separator,
                compression=self.csv_config.compression,
            )
        except FileNotFoundError:
            raise UnableToFindFileException(path)
        except HTTPStatusError:
            raise UnableToFindFileException(path)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        path = self.path.as_posix()
        if path.startswith("http"):
            from io import BytesIO

            buffer = await HttpStorage().read(path)
            io_buffer = BytesIO(buffer)
            io_buffer.seek(0)
            return pl.read_csv(io_buffer, separator=self.csv_config.separator).lazy()

        if not do_file_exist(path):
            raise UnableToFindFileException(path)

        try:
            schema: dict[str, "pl.DataType"] | None = None
            if self.expected_schema:
                schema = {  # type: ignore
                    name: dtype.polars_type
                    for name, dtype in self.expected_schema.items()
                    if not dtype.is_datetime and dtype.name != "bool"
                }

                if self.mapping_keys:
                    reverse_mapping = {v: k for k, v in self.mapping_keys.items()}
                    schema = {
                        reverse_mapping.get(name, name): dtype
                        for name, dtype in schema.items()
                    }

            return pl.scan_csv(
                path, schema_overrides=schema, separator=self.csv_config.separator
            )
        except OSError:
            raise UnableToFindFileException(path)

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        data = (await job.to_lazy_polars()).select(request.all_returned_columns)
        potential_timestamps = request.all_features

        if request.event_timestamp:
            potential_timestamps.add(request.event_timestamp.as_feature())

        for feature in potential_timestamps:
            if feature.dtype.is_datetime:
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

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        data = (await job.to_lazy_polars()).select(request.all_returned_columns)
        for feature in request.features:
            if feature.dtype.is_datetime:
                data = data.with_columns(self.formatter.encode_polars(feature.name))

        if request.event_timestamp:
            data = data.with_columns(
                self.formatter.encode_polars(request.event_timestamp.name)
            )

        if self.mapping_keys:
            names = data.collect_schema().names()
            columns = self.feature_identifier_for(names)
            data = data.rename(dict(zip(names, columns)))

        try:
            existing_df = await self.to_lazy_polars()
            write_df = pl.concat(
                [data, existing_df.select(data.collect_schema().names())],
                how="vertical_relaxed",
            )
        except UnableToFindFileException:
            write_df = data

        await self.write_polars(write_df)

    async def overwrite(
        self,
        job: RetrievalJob,
        request: RetrievalRequest,
        predicate: Expression | None = None,
    ) -> None:
        data = (await job.to_lazy_polars()).select(request.all_returned_columns)

        if predicate is not None:
            polars_exp = predicate.to_polars()
            assert polars_exp is not None
            existing = await self.to_lazy_polars()
            data = pl.concat(
                [data, existing.filter(polars_exp).select(data.columns)], how="vertical"
            )

        for feature in request.features:
            if feature.dtype.is_datetime:
                data = data.with_columns(self.formatter.encode_polars(feature.name))

        if request.event_timestamp:
            data = data.with_columns(
                self.formatter.encode_polars(request.event_timestamp.name)
            )

        if self.mapping_keys:
            columns = self.feature_identifier_for(data.columns)
            data = data.rename(dict(zip(data.columns, columns)))

        logger.error(f"Overwriting {self.path} with {data.columns}")
        await self.write_polars(data)

    async def write_pandas(self, df: pd.DataFrame) -> None:
        create_parent_dir(self.path.as_posix())
        df.to_csv(
            self.path.as_posix(),
            sep=self.csv_config.separator,
            compression=self.csv_config.compression,
            index=self.csv_config.should_write_index,
        )

    async def write_polars(self, df: pl.LazyFrame) -> None:
        create_parent_dir(self.path.as_posix())
        if self.csv_config.compression == "infer":
            df.collect().write_csv(
                self.path.as_posix(),
                separator=self.csv_config.separator,
            )
        else:
            await self.write_pandas(df.collect().to_pandas())

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        with_schema = CsvFileSource(
            path=self.path,
            mapping_keys=self.mapping_keys,
            csv_config=self.csv_config,
            formatter=self.formatter,
            expected_schema={
                feat.name: feat.dtype
                for feat in request.features.union(request.entities)
                if (feat.default_value is None) and not feat.name.isdigit()
            },
        )
        return FileFullJob(with_schema, request, limit, date_formatter=self.formatter)

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        return FileDateJob(
            source=self,
            request=request,
            start_date=start_date,
            end_date=end_date,
            date_formatter=self.formatter,
        )

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls, facts: RetrievalJob, requests: list[tuple[CsvFileSource, RetrievalRequest]]
    ) -> RetrievalJob:
        sources = {source for source, _ in requests}
        if len(sources) != 1:
            raise ValueError(f"Only able to load one {requests} at a time")

        source = list(sources)[0]
        if not isinstance(source, cls):
            raise ValueError(f"Only {cls} is supported, received: {source}")

        # Group based on config
        return FileFactualJob(
            source=source,
            requests=[request for _, request in requests],
            facts=facts,
            date_formatter=source.formatter,
        )

    async def schema(self) -> dict[str, FeatureType]:
        df = await self.to_lazy_polars()
        return {
            name: FeatureType.from_polars(pl_type)
            for name, pl_type in df.schema.items()
        }

    async def feature_view_code(self, view_name: str) -> str:
        from aligned.feature_view.feature_view import FeatureView

        raw_schema = await self.schema()
        schema = {name: feat.feature_factory for name, feat in raw_schema.items()}
        data_source_code = (
            f'FileSource.csv_at("{self.path}", csv_config={self.csv_config})'
        )
        return FeatureView.feature_view_code_template(
            schema,
            data_source_code,
            view_name,
            "from aligned import FileSource\nfrom aligned.sources.local import CsvConfig",
        )

    async def freshness(self, feature: Feature) -> datetime | None:
        return await data_file_freshness(self, feature.name, self.formatter)


@dataclass
class ParquetConfig(Codable):
    """
    A config for how a CSV file should be loaded
    """

    engine: Literal["auto", "pyarrow", "fastparquet"] = field(default="auto")
    compression: Literal["snappy", "gzip", "brotli"] = field(default="snappy")


@dataclass
class PartitionedParquetFileSource(
    CodableBatchDataSource,
    ColumnFeatureMappable,
    DataFileReference,
    WritableFeatureSource,
    Deletable,
):
    """
    A source pointing to a Parquet file
    """

    directory: PathResolver
    partition_keys: list[str]
    mapping_keys: dict[str, str] = field(default_factory=dict)
    config: ParquetConfig = field(default_factory=ParquetConfig)
    date_formatter: DateFormatter = field(default_factory=lambda: DateFormatter.noop())

    type_name: str = "partition_parquet"

    @property
    def as_markdown(self) -> str:
        return f"""#### Partitioned Parquet File
*Partition keys*: {self.partition_keys}

*Renames*: {self.mapping_keys}

*Directory*: {self.directory.as_posix()}

[Go to directory]({self.directory.as_posix()})"""  # noqa

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.directory}"

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    def with_view(self, view: CompiledFeatureView) -> PartitionedParquetFileSource:
        schema_hash = view.schema_hash()
        return PartitionedParquetFileSource(
            directory=self.directory.replace(
                FileDirectory.schema_placeholder(), schema_hash.hex()
            ),
            partition_keys=self.partition_keys,
            mapping_keys=self.mapping_keys,
            config=self.config,
            date_formatter=self.date_formatter,
        )

    async def delete(self, predicate: Expression | None = None) -> None:
        if not predicate:
            delete_path(self.directory.as_posix())
        else:
            polars_exp = predicate.to_polars()
            assert polars_exp is not None
            df = await self.to_lazy_polars()
            filtered_df = df.filter(polars_exp.not_())
            await self.write_polars(filtered_df)

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        glob_path = f"{self.directory}/**/*.parquet"
        try:
            return pl.scan_parquet(glob_path, retries=3, hive_partitioning=True)
        except (OSError, FileNotFoundError):
            raise UnableToFindFileException(glob_path)

    async def write_polars(self, df: pl.LazyFrame) -> None:
        create_parent_dir(self.directory.as_posix())
        df.collect().write_parquet(
            self.directory.as_posix(),
            compression=self.config.compression,
            use_pyarrow=True,
            pyarrow_options={
                "partition_cols": self.partition_keys,
            },
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return FileFullJob(self, request, limit, date_formatter=self.date_formatter)

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        return FileDateJob(
            source=self,
            request=request,
            start_date=start_date,
            end_date=end_date,
            date_formatter=self.date_formatter,
        )

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls,
        facts: RetrievalJob,
        requests: list[tuple[ParquetFileSource, RetrievalRequest]],
    ) -> RetrievalJob:
        from aligned.data_source.batch_data_source import CustomMethodDataSource

        assert len(requests) == 1

        source, request = requests[0]
        if not isinstance(source, cls):
            raise ValueError(f"Only {cls} is supported, received: {source}")

        async def features_for(
            facts: RetrievalJob, request: RetrievalRequest
        ) -> pl.LazyFrame:
            facts_df = await facts.to_lazy_polars()
            partitions = (
                await facts_df.unique(source.partition_keys)
                .select(source.partition_keys)
                .collect_async()
            )

            filter_pl = pl.lit(False)
            for partition in partitions.iter_rows(named=True):
                partition_expr: pl.Expr = pl.lit(True)
                for key, value in partition.items():
                    partition_expr = partition_expr & (pl.col(key) == value)

                filter_pl = filter_pl | partition_expr

            source_values = await source.to_lazy_polars()
            filtered_values = source_values.filter(filter_pl)

            return await FileFactualJob(
                RetrievalJob.from_polars_df(filtered_values, [request]),
                requests=[request],
                facts=RetrievalJob.from_polars_df(facts_df, facts.retrieval_requests),
            ).to_lazy_polars()

        return CustomMethodDataSource.from_methods(
            features_for=features_for,
        ).features_for(facts, request)

    async def schema(self) -> dict[str, FeatureType]:
        glob_path = f"{self.directory}/**/*.parquet"
        parquet_schema = pl.scan_parquet(glob_path).schema
        return {
            name: FeatureType.from_polars(pl_type)
            for name, pl_type in parquet_schema.items()
        }

    async def feature_view_code(self, view_name: str) -> str:
        from aligned.feature_view.feature_view import FeatureView

        raw_schema = await self.schema()
        schema = {name: feat.feature_factory for name, feat in raw_schema.items()}
        data_source_code = f'FileSource.partitioned_parquet_at("{self.directory}", {self.partition_keys})'
        return FeatureView.feature_view_code_template(
            schema, data_source_code, view_name, "from aligned import FileSource"
        )

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        job = job.select(request.all_returned_columns)
        df = await job.to_lazy_polars()
        await self.write_polars(df)

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        import shutil

        upsert_on = sorted(request.entity_names)

        returned_columns = request.all_returned_columns
        df = await job.select(returned_columns).to_polars()
        unique_partitions = df.select(self.partition_keys).unique()

        final_filter: pl.Expr | None = None
        for row in unique_partitions.iter_rows(named=True):
            current: pl.Expr | None = None

            for key, value in row.items():
                if current is not None:
                    current = current & (pl.col(key) == value)
                else:
                    current = pl.col(key) == value

            if current is not None:
                if final_filter is not None:
                    final_filter = final_filter | current
                else:
                    final_filter = current

        assert final_filter is not None, "Found partitions to filter on"
        try:
            existing_df = (await self.to_lazy_polars()).filter(final_filter)
            write_df = (
                upsert_on_column(upsert_on, df.lazy(), existing_df)
                .select(returned_columns)
                .collect()
            )
        except (UnableToFindFileException, pl.exceptions.ComputeError):
            write_df = df.lazy()

        for row in unique_partitions.iter_rows(named=True):
            dir = Path(self.directory.as_posix())
            for partition_key in self.partition_keys:
                dir = dir / f"{partition_key}={row[partition_key]}"

            if dir.exists():
                shutil.rmtree(dir.as_posix())

        await self.write_polars(write_df.lazy())

    async def overwrite(
        self,
        job: RetrievalJob,
        request: RetrievalRequest,
        predicate: Expression | None = None,
    ) -> None:
        import shutil

        if predicate:
            raise NotImplementedError(
                f"Have not implemented the partial overwrite for {type(self)}"
            )

        posix_dir = self.directory.as_posix()
        if Path(posix_dir).exists():
            shutil.rmtree(posix_dir)

        await self.insert(job, request)


@dataclass
class ParquetFileSource(
    CodableBatchDataSource, ColumnFeatureMappable, DataFileReference, Deletable
):
    """
    A source pointing to a Parquet file
    """

    path: PathResolver
    mapping_keys: dict[str, str] = field(default_factory=dict)
    config: ParquetConfig = field(default_factory=ParquetConfig)
    date_formatter: DateFormatter = field(default_factory=lambda: DateFormatter.noop())

    type_name: str = "parquet"

    @property
    def as_markdown(self) -> str:
        return f"""#### Parquet File
*Renames*: {self.mapping_keys}

*File*: {self.path.as_posix()}

[Go to file]({self.path.as_posix()})"""  # noqa

    def with_view(self, view: CompiledFeatureView) -> ParquetFileSource:
        schema_hash = view.schema_hash()
        return ParquetFileSource(
            path=self.path.replace(
                FileDirectory.schema_placeholder(), schema_hash.hex()
            ),
            mapping_keys=self.mapping_keys,
            config=self.config,
            date_formatter=self.date_formatter,
        )

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path}"

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    async def delete(self, predicate: Expression | None = None) -> None:
        if not predicate:
            delete_path(self.path.as_posix())
        else:
            polars_exp = predicate.to_polars()
            assert polars_exp is not None
            df = await self.to_lazy_polars()
            filtered_df = df.filter(polars_exp.not_())
            await self.write_polars(filtered_df)

    async def read_pandas(self) -> pd.DataFrame:
        path = self.path.as_posix()
        try:
            return pd.read_parquet(path)
        except FileNotFoundError:
            raise UnableToFindFileException(path)
        except HTTPStatusError:
            raise UnableToFindFileException(path)

    async def write_pandas(self, df: pd.DataFrame) -> None:
        create_parent_dir(self.path.as_posix())
        df.to_parquet(
            self.path.as_posix(),
            engine=self.config.engine,
            compression=self.config.compression,
            index=False,
        )

    async def to_lazy_polars(self) -> pl.LazyFrame:
        path = self.path.as_posix()
        if (not path.startswith("http")) and (not do_file_exist(path)):
            raise UnableToFindFileException(path)

        try:
            return pl.scan_parquet(path)
        except OSError:
            raise UnableToFindFileException(path)

    async def write_polars(self, df: pl.LazyFrame) -> None:
        path = self.path.as_posix()
        create_parent_dir(path)
        df.collect().write_parquet(path, compression=self.config.compression)

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return FileFullJob(self, request, limit, date_formatter=self.date_formatter)

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        return FileDateJob(
            source=self,
            request=request,
            start_date=start_date,
            end_date=end_date,
            date_formatter=self.date_formatter,
        )

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls,
        facts: RetrievalJob,
        requests: list[tuple[ParquetFileSource, RetrievalRequest]],
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

    async def schema(self) -> dict[str, FeatureType]:
        path = self.path.as_posix()
        if path.startswith("http"):
            parquet_schema = pl.scan_parquet(path).schema
        else:
            parquet_schema = pl.read_parquet_schema(path)

        return {
            name: FeatureType.from_polars(pl_type)
            for name, pl_type in parquet_schema.items()
        }

    async def feature_view_code(self, view_name: str) -> str:
        from aligned.feature_view.feature_view import FeatureView

        raw_schema = await self.schema()
        schema = {name: feat.feature_factory for name, feat in raw_schema.items()}
        data_source_code = f'FileSource.parquet_at("{self.path}")'
        return FeatureView.feature_view_code_template(
            schema, data_source_code, view_name, "from aligned import FileSource"
        )


@dataclass
class DeltaFileConfig(Codable):
    mode: Literal["append", "overwrite", "error"] = field(default="append")
    overwrite_schema: bool = field(default=False)


@dataclass
class DeltaConfig(Codable):
    schema_mode: Literal["merge", "overwrite"] | None = field(default=None)
    partition_by: list[str] | str | None = field(default=None)
    target_file_size: int | None = field(default=None)

    def write_options(self) -> dict[str, Any]:
        return {
            key: value for key, value in self.to_dict().items() if value is not None
        }


@dataclass
class DeltaFileSource(
    CodableBatchDataSource,
    ColumnFeatureMappable,
    DataFileReference,
    WritableFeatureSource,
    Deletable,
):
    """
    A source pointing to a Parquet file
    """

    path: PathResolver
    mapping_keys: dict[str, str] = field(default_factory=dict)
    config: DeltaConfig = field(default_factory=DeltaConfig)
    date_formatter: DateFormatter = field(default_factory=lambda: DateFormatter.noop())

    azure_config: AzureBlobConfig | None = field(default=None)

    type_name: str = "delta"

    def storage_options(self) -> dict[str, Any] | None:
        if self.azure_config:
            return self.azure_config.read_creds()
        return None

    def resolved_path(self) -> str:
        raw_path = self.path.as_posix()
        if self.azure_config:
            return f"az://{raw_path}"
        return raw_path

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path}"

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    async def delete(self, predicate: Expression | None = None) -> None:
        if not predicate:
            delete_path(self.path.as_posix())
            return

    async def read_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    async def write_pandas(self, df: pd.DataFrame) -> None:
        await self.write_polars(pl.from_pandas(df).lazy())

    async def to_lazy_polars(self) -> pl.LazyFrame:
        storage_options = self.storage_options()

        if storage_options is None and not do_dir_exist(self.path.as_posix()):
            raise UnableToFindFileException(self.path.as_posix())

        try:
            return pl.scan_delta(self.resolved_path(), storage_options=storage_options)
        except OSError:
            raise UnableToFindFileException(self.resolved_path())

    async def write_polars(self, df: pl.LazyFrame) -> None:
        storage_options = self.storage_options()

        if storage_options is None:
            # Only in local dirs
            create_parent_dir(self.path.as_posix())

        df.collect().write_delta(
            self.resolved_path(),
            delta_write_options=self.config.write_options(),
            storage_options=storage_options,
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return FileFullJob(self, request, limit, date_formatter=self.date_formatter)

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        return FileDateJob(
            source=self,
            request=request,
            start_date=start_date,
            end_date=end_date,
            date_formatter=self.date_formatter,
        )

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls,
        facts: RetrievalJob,
        requests: list[tuple[DeltaFileSource, RetrievalRequest]],
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

    async def schema(self) -> dict[str, FeatureType]:
        parquet_schema = pl.read_delta(
            self.resolved_path(), storage_options=self.storage_options()
        ).schema
        return {
            name: FeatureType.from_polars(pl_type)
            for name, pl_type in parquet_schema.items()
        }

    async def feature_view_code(self, view_name: str) -> str:
        from aligned.feature_view.feature_view import FeatureView

        raw_schema = await self.schema()
        schema = {name: feat.feature_factory for name, feat in raw_schema.items()}
        data_source_code = f'FileSource.parquet_at("{self.path}")'
        return FeatureView.feature_view_code_template(
            schema, data_source_code, view_name, "from aligned import FileSource"
        )

    async def overwrite(
        self,
        job: RetrievalJob,
        request: RetrievalRequest,
        predicate: Expression | None = None,
    ) -> None:
        write_options = self.config.write_options()

        if predicate:
            glot = predicate.to_glot()
            assert glot is not None
            write_options["predicate"] = glot.sql(dialect="spark")

        storage_options = self.storage_options()

        data = await job.to_lazy_polars()
        data.select(request.all_returned_columns).collect().write_delta(
            self.resolved_path(),
            mode="overwrite",
            delta_write_options=write_options,
            storage_options=storage_options,
        )

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        data = await job.to_lazy_polars()
        data.select(request.all_returned_columns).collect().write_delta(
            self.resolved_path(),
            mode="append",
            delta_write_options=self.config.write_options(),
            storage_options=self.storage_options(),
        )

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        new_data = await job.to_lazy_polars()
        existing = await self.to_lazy_polars()

        # Should to a merge statement instead
        upsert_on_column(
            list(request.entity_names), new_data, existing
        ).collect().write_delta(
            self.resolved_path(),
            mode="overwrite",
            storage_options=self.storage_options(),
            delta_write_options=self.config.write_options(),
        )


@dataclass
class StorageFileSource(StorageFileReference, Codable):
    path: PathResolver
    azure_config: AzureBlobConfig | None = field(default=None)

    def posix_path(self) -> str:
        return self.path.as_posix()

    @property
    def storage(self) -> Storage:
        if self.azure_config:
            return self.azure_config.storage

        path = self.path.as_posix()
        if path.startswith("http"):
            return HttpStorage()
        else:
            return FileStorage()

    def __hash__(self) -> int:
        return hash(self.path)

    async def read(self) -> bytes:
        return await self.storage.read(self.path.as_posix())

    async def write(self, content: bytes | bytearray) -> None:
        await self.storage.write(self.path.as_posix(), content)


@dataclass
class DirectoryRepo(AsRepoDefinition):
    dir: Path
    exclude: list[str] | None = field(default=None)

    async def as_repo_definition(self) -> RepoDefinition:
        from aligned.compiler.repo_reader import RepoReader

        return await RepoReader.definition_from_path(self.dir, self.exclude)


class Directory(Protocol):
    def json_at(self, path: str) -> StorageFileReference: ...

    def csv_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        csv_config: CsvConfig | None = None,
    ) -> CodableBatchDataSource: ...

    def partitioned_parquet_at(
        self,
        directory: str,
        partition_keys: list[str],
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> CodableBatchDataSource: ...

    def parquet_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> CodableBatchDataSource: ...

    def delta_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: DeltaConfig | None = None,
    ) -> CodableBatchDataSource: ...

    def sub_directory(self, path: str | ConfigValue) -> Directory: ...

    def with_schema_version(
        self, sub_directory: str | ConfigValue | None = None
    ) -> Directory: ...


@dataclass
class FileDirectory(Codable, Directory):
    path: PathResolver

    @classmethod
    def schema_placeholder(cls) -> PlaceholderValue:
        return PlaceholderValue("schema_version_placeholder")

    def json_at(self, path: str) -> StorageFileSource:
        return StorageFileSource(path=self.path.append(path))

    def csv_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        csv_config: CsvConfig | None = None,
    ) -> CsvFileSource:
        return CsvFileSource(
            self.path.append(path),
            mapping_keys=mapping_keys or {},
            csv_config=csv_config or CsvConfig(),
        )

    def parquet_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> ParquetFileSource:
        return ParquetFileSource(
            path=self.path.append(path),
            mapping_keys=mapping_keys or {},
            config=config or ParquetConfig(),
            date_formatter=date_formatter or DateFormatter.noop(),
        )

    def partitioned_parquet_at(
        self,
        directory: str,
        partition_keys: list[str],
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> PartitionedParquetFileSource:
        return PartitionedParquetFileSource(
            directory=self.path.append(directory),
            partition_keys=partition_keys,
            mapping_keys=mapping_keys or {},
            config=config or ParquetConfig(),
            date_formatter=date_formatter or DateFormatter.noop(),
        )

    def delta_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: DeltaConfig | None = None,
    ) -> DeltaFileSource:
        return DeltaFileSource(
            PathResolver([ConfigValue.from_value(path)]),
            mapping_keys or {},
            config=config or DeltaConfig(),
        )

    def sub_directory(self, path: str | ConfigValue) -> FileDirectory:
        return FileDirectory(self.path.append(path))

    def with_schema_version(
        self, sub_directory: str | ConfigValue | None = None
    ) -> Directory:
        if sub_directory:
            return FileDirectory(
                self.path.append(sub_directory).append(
                    FileDirectory.schema_placeholder()
                )
            )
        else:
            return FileDirectory(self.path.append(FileDirectory.schema_placeholder()))

    def directory(self, path: str | ConfigValue) -> FileDirectory:
        return self.sub_directory(path)

    def repo_from_dir(
        self, dir: str, exclude: list[str] | None = None
    ) -> AsRepoDefinition:
        return DirectoryRepo(Path(dir), exclude)


class FileSource:
    """
    A factory class, creating references to files.

    This therefore abstracts away the concrete classes the users wants.
    Therefore making them easier to discover.
    """

    @staticmethod
    def json_at(path: str) -> StorageFileSource:
        return StorageFileSource(path=PathResolver.from_value(path))

    @staticmethod
    def csv_at(
        path: str,
        mapping_keys: dict[str, str] | None = None,
        csv_config: CsvConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> CsvFileSource:
        return CsvFileSource(
            PathResolver.from_value(path),
            mapping_keys=mapping_keys or {},
            csv_config=csv_config or CsvConfig(),
            formatter=date_formatter or DateFormatter.iso_8601(),
        )

    @staticmethod
    def partitioned_parquet_at(
        directory: str,
        partition_keys: list[str],
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> PartitionedParquetFileSource:
        return PartitionedParquetFileSource(
            directory=PathResolver.from_value(directory),
            partition_keys=partition_keys,
            mapping_keys=mapping_keys or {},
            config=config or ParquetConfig(),
            date_formatter=date_formatter or DateFormatter.noop(),
        )

    @staticmethod
    def parquet_at(
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> ParquetFileSource:
        return ParquetFileSource(
            path=PathResolver.from_value(path),
            mapping_keys=mapping_keys or {},
            config=config or ParquetConfig(),
            date_formatter=date_formatter or DateFormatter.noop(),
        )

    @staticmethod
    def delta_at(
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: DeltaConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> DeltaFileSource:
        return DeltaFileSource(
            PathResolver([ConfigValue.from_value(path)]),
            mapping_keys or {},
            config=config or DeltaConfig(),
            date_formatter=date_formatter or DateFormatter.noop(),
        )

    @staticmethod
    def directory(path: str | ConfigValue) -> FileDirectory:
        return FileDirectory(PathResolver.from_value(path))

    def with_schema_version(
        self, sub_directory: str | ConfigValue | None = None
    ) -> Directory:
        if sub_directory:
            return FileDirectory(
                PathResolver.from_value(sub_directory).append(
                    FileDirectory.schema_placeholder()
                )
            )
        else:
            return FileDirectory(
                PathResolver.from_value(".").append(FileDirectory.schema_placeholder())
            )

    @staticmethod
    def repo_from_dir(dir: str, exclude: list[str] | None = None) -> AsRepoDefinition:
        return DirectoryRepo(Path(dir), exclude)


class LiteralReference(DataFileReference):
    """
    A class containing a in mem pandas frame.

    This makes it easier standardise the interface when writing data.
    """

    file: pl.LazyFrame

    def __init__(self, file: pl.LazyFrame | pd.DataFrame | pl.DataFrame) -> None:
        if isinstance(file, pl.DataFrame):
            self.file = file.lazy()
        elif isinstance(file, pl.LazyFrame):
            self.file = file
        elif isinstance(file, pd.DataFrame):
            self.file = pl.from_pandas(file).lazy()
        else:
            raise ValueError(f"Unsupported type {type(file)}")

    def job_group_key(self) -> str:
        return str(uuid4())

    async def read_pandas(self) -> pd.DataFrame:
        return self.file.collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return self.file
