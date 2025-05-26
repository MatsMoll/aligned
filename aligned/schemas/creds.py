from __future__ import annotations

from aligned.schemas.codable import Codable
from aligned.config_value import ConfigValue

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass


class AzureCreds(Codable):
    account_id: ConfigValue
    tenant_id: ConfigValue
    client_id: ConfigValue
    client_secret: ConfigValue
    account_name: ConfigValue

    def storage_options(self) -> dict[str, str]:
        return {
            "account_id": self.account_id.read(),
            "tenant_id": self.tenant_id.read(),
            "client_id": self.client_id.read(),
            "client_secret": self.client_secret.read(),
            "account_name": self.account_name.read(),
        }


class AwsCreds:
    pass


class Credentials(Codable):
    azure: AzureCreds | None = None
    aws: AwsCreds | None = None

    def file_system(self) -> str:
        if self.azure:
            return "az://"
        elif self.aws:
            return "s3://"
        else:
            return "file://"

    def storage_options(self) -> dict[str, str] | None:
        if self.azure:
            return self.storage_options()
        return None


# @dataclass
# class DeltaFileSource(
#     CodableBatchDataSource,
#     ColumnFeatureMappable,
#     DataFileReference,
#     WritableFeatureSource,
# ):
#     """
#     A source pointing to a Delta file
#     """
#
#     path: PathResolver
#     mapping_keys: dict[str, str] = field(default_factory=dict)
#     config: DeltaFileConfig = field(default_factory=DeltaFileConfig)
#     date_formatter: DateFormatter = field(default_factory=lambda: DateFormatter.noop())
#     creds: Credentials = field(default_factory=Credentials)
#
#     type_name: str = "delta"
#
#     def job_group_key(self) -> str:
#         return f"{self.type_name}/{self.path}"
#
#     def __hash__(self) -> int:
#         return hash(self.job_group_key())
#
#     async def delete(self) -> None:
#         delete_path(self.path)
#
#     async def read_pandas(self) -> pd.DataFrame:
#         return (await self.to_lazy_polars()).collect().to_pandas()
#
#     async def write_pandas(self, df: pd.DataFrame) -> None:
#         await self.write_polars(pl.from_pandas(df).lazy())
#
#     async def to_lazy_polars(self) -> pl.LazyFrame:
#         if not do_dir_exist(self.path):
#             raise UnableToFindFileException(self.path.as_posix())
#
#         try:
#             return pl.scan_delta(self.path.as_posix())
#         except OSError:
#             raise UnableToFindFileException(self.path.as_posix())
#
#     async def write_polars(self, df: pl.LazyFrame) -> None:
#         create_parent_dir(self.path)
#         df.collect().write_delta(
#             self.path.as_posix(),
#             mode=self.config.mode,
#             overwrite_schema=self.config.overwrite_schema,
#         )
#
#     def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
#         return FileFullJob(self, request, limit, date_formatter=self.date_formatter)
#
#     def all_between_dates(
#         self, request: RetrievalRequest, start_date: datetime, end_date: datetime
#     ) -> RetrievalJob:
#         return FileDateJob(
#             source=self,
#             request=request,
#             start_date=start_date,
#             end_date=end_date,
#             date_formatter=self.date_formatter,
#         )
#
#     @classmethod
#     def multi_source_features_for(  # type: ignore
#         cls,
#         facts: RetrievalJob,
#         requests: list[tuple[DeltaFileSource, RetrievalRequest]],
#     ) -> RetrievalJob:
#         source = requests[0][0]
#         if not isinstance(source, cls):
#             raise ValueError(f"Only {cls} is supported, received: {source}")
#
#         # Group based on config
#         return FileFactualJob(
#             source=source,
#             requests=[request for _, request in requests],
#             facts=facts,
#             date_formatter=source.date_formatter,
#         )
#
#     async def schema(self) -> dict[str, FeatureType]:
#         parquet_schema = pl.read_delta(self.path.as_posix()).schema
#         return {
#             name: FeatureType.from_polars(pl_type)
#             for name, pl_type in parquet_schema.items()
#         }
#
#     async def feature_view_code(self, view_name: str) -> str:
#         from aligned.feature_view.feature_view import FeatureView
#
#         raw_schema = await self.schema()
#         schema = {name: feat.feature_factory for name, feat in raw_schema.items()}
#         data_source_code = f'FileSource.parquet_at("{self.path}")'
#         return FeatureView.feature_view_code_template(
#             schema, data_source_code, view_name, "from aligned import FileSource"
#         )
#
#     async def overwrite(self, job: RetrievalJob, request: RetrievalRequest) -> None:
#         data = await job.to_lazy_polars()
#         data.select(request.all_returned_columns).collect().write_delta(
#             self.path.as_posix(), mode="overwrite"
#         )
#
#     async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
#         data = await job.to_lazy_polars()
#         data.select(request.all_returned_columns).collect().write_delta(
#             self.path.as_posix(), mode="append", storage_options=self.creds.storage_options()
#         )
#
#     async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
#         new_data = await job.to_lazy_polars()
#         existing = await self.to_lazy_polars()
#
#         # Should to a merge statement instead
#         upsert_on_column(
#             list(request.entity_names), new_data, existing
#         ).collect().write_delta(
#             self.path.as_posix(), mode="overwrite", storage_options=self.creds.storage_options()
#         )
#
#
