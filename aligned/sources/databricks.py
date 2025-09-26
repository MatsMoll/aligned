from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from aligned.data_source.batch_data_source import (
    BatchDataSource,
    CodableBatchDataSource,
    FeatureType,
    RequestResult,
)
from aligned.feature_source import WritableFeatureSource
from aligned.retrieval_job import FilterRepresentable, RetrievalJob, RetrievalRequest
from aligned.schemas.constraints import MaxLength, MinLength
from aligned.schemas.feature import Constraint, Feature, FeatureLocation
from aligned.schemas.transformation import (
    Expression,
)
from aligned.sources.local import FileFactualJob
from aligned.config_value import EnvironmentValue, LiteralValue, ConfigValue
from aligned.lazy_imports import databricks_fe
from aligned.sources.renamer import Renamer

if TYPE_CHECKING:
    from sqlglot import exp
    import pandas as pd
    from pyspark.sql import SparkSession, DataFrame as SparkFrame
    from pyspark.sql.types import DataType, StructType


logger = logging.getLogger(__name__)


@dataclass
class DatabricksAuthConfig:
    token: str
    host: str


def raise_on_invalid_pyspark_schema(schema: DataType) -> None:
    from pyspark.sql.types import ArrayType, NullType, StructType

    if isinstance(schema, StructType):
        for field in schema.fields:
            raise_on_invalid_pyspark_schema(field)

    if isinstance(schema, ArrayType):
        raise_on_invalid_pyspark_schema(schema.elementType)

    if isinstance(schema, NullType):
        raise ValueError("Found a NullType in the schema. This will lead to issues.")


def polars_dtype_to_spark(data_type: pl.PolarsDataType) -> DataType:  # noqa: PLR0911
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        ByteType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    if isinstance(data_type, pl.String):
        return StringType()
    if isinstance(data_type, pl.Float32):
        return FloatType()
    if isinstance(data_type, pl.Float64):
        return DoubleType()
    if isinstance(data_type, pl.Int8):
        return ByteType()
    if isinstance(data_type, pl.Int16):
        return ShortType()
    if isinstance(data_type, pl.Int32):
        return IntegerType()
    if isinstance(data_type, pl.Int64):
        return LongType()
    if isinstance(data_type, pl.Boolean):
        return BooleanType()
    if isinstance(data_type, pl.Datetime):
        return TimestampType()
    if isinstance(data_type, (pl.Array, pl.List)):
        if data_type.inner:
            return ArrayType(polars_dtype_to_spark(data_type.inner))
        return ArrayType(StringType())
    if isinstance(data_type, pl.Struct):
        return StructType(
            [
                StructField(
                    name=field.name, dataType=polars_dtype_to_spark(field.dtype)
                )
                for field in data_type.fields
            ]
        )

    raise ValueError(f"Unsupported type {data_type}")


@dataclass
class SparkDataType:
    dtype: FeatureType
    constraints: list[Constraint]


def convert_pyspark_type(data_type: DataType) -> SparkDataType:  # noqa: PLR0911
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        ByteType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        MapType,
        ShortType,
        StringType,
        StructType,
        TimestampNTZType,
        TimestampType,
        CharType,
        DateType,
        VarcharType,
    )

    def no_constraints(dtype: FeatureType) -> SparkDataType:
        return SparkDataType(dtype, [])

    if isinstance(data_type, VarcharType):
        return SparkDataType(
            FeatureType.string(),
            [
                MaxLength(data_type.length),
            ],
        )
    if isinstance(data_type, CharType):
        return SparkDataType(
            FeatureType.string(),
            [
                MinLength(data_type.length),
                MaxLength(data_type.length),
            ],
        )
    if isinstance(data_type, StringType):
        return no_constraints(FeatureType.string())
    if isinstance(data_type, FloatType):
        return no_constraints(FeatureType.floating_point())
    if isinstance(data_type, DoubleType):
        return no_constraints(FeatureType.double())
    if isinstance(data_type, ByteType):
        return no_constraints(FeatureType.int8())
    if isinstance(data_type, ShortType):
        return no_constraints(FeatureType.int16())
    if isinstance(data_type, IntegerType):
        return no_constraints(FeatureType.int32())
    if isinstance(data_type, LongType):
        return no_constraints(FeatureType.int64())
    if isinstance(data_type, BooleanType):
        return no_constraints(FeatureType.boolean())
    if isinstance(data_type, (TimestampType, TimestampNTZType)):
        return no_constraints(FeatureType.datetime())
    if isinstance(data_type, DateType):
        return no_constraints(FeatureType.date())
    if isinstance(data_type, ArrayType):
        return no_constraints(
            FeatureType.array(convert_pyspark_type(data_type.elementType).dtype)
        )
    if isinstance(data_type, StructType):
        return no_constraints(FeatureType.json())
    if isinstance(data_type, MapType):
        return no_constraints(FeatureType.json())

    raise ValueError(f"Unsupported type {data_type}")


@dataclass(init=False)
class DatabricksConnectionConfig:
    host: ConfigValue
    cluster_id: ConfigValue | None
    token: ConfigValue | None

    azure_client_id: ConfigValue | None = None
    azure_client_secret: ConfigValue | None = None
    azure_tenant_id: ConfigValue | None = None

    def __init__(
        self,
        host: str | ConfigValue,
        cluster_id: str | ConfigValue | None,
        token: str | ConfigValue | None,
        azure_client_id: ConfigValue | None = None,
        azure_client_secret: ConfigValue | None = None,
        azure_tenant_id: ConfigValue | None = None,
    ) -> None:
        self.host = LiteralValue.from_value(host)
        self.cluster_id = LiteralValue.from_value(cluster_id) if cluster_id else None
        self.token = LiteralValue.from_value(token) if token else None
        self.azure_client_secret = azure_client_secret
        self.azure_client_id = azure_client_id
        self.azure_tenant_id = azure_tenant_id

    def storage_provider(self) -> pl.CredentialProvider | None:
        if self.azure_client_id and self.azure_client_secret and self.azure_tenant_id:
            from azure.identity import ClientSecretCredential

            creds = ClientSecretCredential(
                tenant_id=self.azure_tenant_id.read(),
                client_id=self.azure_client_id.read(),
                client_secret=self.azure_client_secret.read(),
            )

            return pl.CredentialProviderAzure(credential=creds)
        return None

    def with_auth(
        self, token: str | ConfigValue, host: str | ConfigValue
    ) -> DatabricksConnectionConfig:
        return DatabricksConnectionConfig(
            cluster_id=self.cluster_id, token=token, host=host
        )

    @staticmethod
    def databricks_or_serverless(
        host: str | ConfigValue | None = None, token: str | ConfigValue | None = None
    ) -> DatabricksConnectionConfig:
        return DatabricksConnectionConfig(
            cluster_id=None,
            token=token or EnvironmentValue("DATABRICKS_TOKEN"),
            host=host or EnvironmentValue("DATABRICKS_HOST"),
        )

    @staticmethod
    def serverless(
        host: str | ConfigValue | None = None, token: str | ConfigValue | None = None
    ) -> DatabricksConnectionConfig:
        return DatabricksConnectionConfig(
            cluster_id="serverless",
            token=token or EnvironmentValue("DATABRICKS_TOKEN"),
            host=host or EnvironmentValue("DATABRICKS_HOST"),
        )

    @staticmethod
    def with_cluster_id(
        cluster_id: str | ConfigValue, host: str | ConfigValue
    ) -> DatabricksConnectionConfig:
        return DatabricksConnectionConfig(cluster_id=cluster_id, token=None, host=host)

    def catalog(self, catalog: str | ConfigValue) -> UnityCatalog:
        return UnityCatalog(self, LiteralValue.from_value(catalog))

    def connection(self) -> SparkSession:
        from pyspark.errors import PySparkException

        cluster_id = self.cluster_id

        if not cluster_id:
            from databricks.sdk.runtime import spark

            if spark is not None:
                return spark

            # If no spark session
            # Assume that serverless is available
            cluster_id = LiteralValue("serverless")

        from databricks.connect.session import DatabricksSession

        builder = DatabricksSession.builder.host(self.host.read())

        cluster_id_value = cluster_id.read()
        if cluster_id_value == "serverless":
            builder = builder.serverless()
        else:
            builder = builder.clusterId(cluster_id_value)

        if self.token:
            builder = builder.token(self.token.read())

        if cluster_id_value == "serverless":
            spark = builder.getOrCreate()
            try:
                spark.sql("SELECT 1")
                return spark
            except PySparkException:
                spark.stop()

        return builder.getOrCreate()

    def sql(self, query: str) -> UCSqlSource:
        """
        Creates a SQL query source by running the defined query.
        """
        return UCSqlSource(self, query)

    def sql_file(
        self, file: str | Path, format_values: dict[str, str] | None = None
    ) -> UCSqlSource:
        """
        Creates a SQL query source by reading the file.
        """
        if not isinstance(file, Path):
            file = Path(file)

        content = file.read_text()

        if format_values:
            content = content.format(**format_values)

        return UCSqlSource(self, content)


@dataclass
class UnityCatalog:
    config: DatabricksConnectionConfig

    catalog: ConfigValue

    def schema(self, schema: str | ConfigValue) -> UnityCatalogSchema:
        return UnityCatalogSchema(
            self.config, self.catalog, LiteralValue.from_value(schema)
        )

    def sql(self, query: str) -> UCSqlSource:
        return UCSqlSource(self.config, query)


@dataclass
class UnityCatalogSchema:
    config: DatabricksConnectionConfig

    catalog: ConfigValue
    schema: ConfigValue

    async def list_tables(self) -> list[str]:
        con = self.config.connection()
        tables = con.sql(
            f"SHOW TABLES {self.catalog.read()}.{self.schema.read()};"
        ).toPandas()
        return tables["tableName"].to_list()

    def table(self, table: str | ConfigValue) -> UCTableSource:
        return UCTableSource(
            self.config,
            UnityCatalogTableConfig(
                self.catalog, self.schema, LiteralValue.from_value(table)
            ),
        )

    def feature_table(self, table: str | ConfigValue) -> UCFeatureTableSource:
        return UCFeatureTableSource(
            self.config,
            UnityCatalogTableConfig(
                self.catalog, self.schema, LiteralValue.from_value(table)
            ),
        )


@dataclass
class UnityCatalogTableConfig:
    catalog: ConfigValue
    schema: ConfigValue
    table: ConfigValue

    def identifier(self) -> str:
        return f"{self.catalog.read()}.{self.schema.read()}.{self.table.read()}"


class DatabricksSource:
    """
    A generic config making it easier to find all sources related to databricks.

    ```python
    store = await ContractStore.from_dir()

    store.sources_of_type(
        DatabricksSource,
        lambda databricks: databricks.source = ...
    )
    ```
    """

    config: DatabricksConnectionConfig


@dataclass
class UCSqlJob(RetrievalJob, DatabricksSource):
    config: DatabricksConnectionConfig
    query: exp.Select
    request: RetrievalRequest

    filter_exp: Expression | None = field(default=None)

    @property
    def request_result(self) -> RequestResult:
        return self.request.request_result

    @property
    def retrieval_requests(self) -> list[RetrievalRequest]:
        return [self.request]

    async def to_spark(self, session: SparkSession | None = None) -> SparkFrame:
        client = session or self.config.connection()

        spark_df = client.sql(self.query.sql(dialect="spark"))

        if self.filter_exp is None:
            return spark_df

        sp_filter = self.filter_exp.to_spark()
        assert (
            sp_filter is not None
        ), f"Unable to create spark filter from '{self.filter_exp}'"
        return spark_df.filter(sp_filter)

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return pl.from_pandas(await self.to_pandas()).lazy()

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_spark()).toPandas()

    def filter(self, condition: FilterRepresentable) -> RetrievalJob:
        if isinstance(condition, str):
            return UCSqlJob(
                config=self.config,
                query=self.query.where(condition, dialect="spark"),
                request=self.request,
                filter_exp=self.filter_exp,
            )

        try:
            exp = Expression.from_value(condition)

            return UCSqlJob(
                config=self.config,
                query=self.query,
                request=self.request,
                filter_exp=self.filter_exp & exp if self.filter_exp else exp,
            )
        except Exception:
            return RetrievalJob.filter(self, condition)


@dataclass
class UCSqlSource(CodableBatchDataSource, DatabricksSource):
    config: DatabricksConnectionConfig
    query: str

    type_name = "uc_sql"

    @property
    def as_markdown(self) -> str:
        return f"""Running SQL query in Databricks {self.config}
```sql
{self.query}
```
"""

    def job_group_key(self) -> str:
        return str(hash(self.query))

    def depends_on(self) -> set[FeatureLocation]:
        from sqlglot import parse_one, exp

        tree = parse_one(self.query, read="spark")

        ctes = tree.find_all(exp.CTE)
        tables = tree.find_all(exp.Table)

        cte_names = {cte.alias_or_name for cte in ctes}

        return {
            FeatureLocation.feature_view(table.name)
            for table in tables
            if table.name not in cte_names
        }

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        from sqlglot import parse_one, exp

        expression = parse_one(self.query, read="spark")

        assert isinstance(
            expression, exp.Select
        ), f"Unable to read a spark query that is not a SELECT. Got {type(expression)}"

        if limit:
            expression = expression.limit(limit)

        return UCSqlJob(self.config, expression, request)

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        raise NotImplementedError(type(self))

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls: type[UCSqlSource],
        facts: RetrievalJob,
        requests: list[tuple[UCSqlSource, RetrievalRequest]],
    ) -> RetrievalJob:
        raise NotImplementedError(cls)

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        return type(self).multi_source_features_for(facts, [(self, request)])

    async def schema(self) -> dict[str, FeatureType]:
        """Returns the schema for the data source

        ```python
        source = FileSource.parquet_at('test_data/titanic.parquet')
        schema = await source.schema()
        >>> {'passenger_id': FeatureType(name='int64'), ...}
        ```

        Returns:
            dict[str, FeatureType]: A dictionary containing the column name and the feature type
        """
        raise NotImplementedError(f"`schema()` is not implemented for {type(self)}.")

    async def freshness(self, feature: Feature) -> datetime | None:
        """
        my_table_freshenss = await (PostgreSQLConfig("DB_URL")
            .table("my_table")
            .freshness()
        )
        """
        raise NotImplementedError(type(self))

    def with_config(self, config: DatabricksConnectionConfig) -> UCSqlSource:
        return UCSqlSource(config, self.query)


@dataclass
class UCFeatureTableSource(
    CodableBatchDataSource, WritableFeatureSource, DatabricksSource
):
    config: DatabricksConnectionConfig
    table: UnityCatalogTableConfig

    type_name = "uc_feature_table"

    def job_group_key(self) -> str:
        return "uc_feature_table"

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        client = databricks_fe.FeatureEngineeringClient()

        async def load() -> pl.LazyFrame:
            spark_df = client.read_table(name=self.table.identifier())

            if limit:
                spark_df = spark_df.limit(limit)

            return pl.from_pandas(spark_df.toPandas()).lazy()

        return RetrievalJob.from_lazy_function(load, request)

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        raise NotImplementedError(type(self))

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls: type[UCFeatureTableSource],
        facts: RetrievalJob,
        requests: list[tuple[UCFeatureTableSource, RetrievalRequest]],
    ) -> RetrievalJob:
        keys = {
            source.job_group_key()
            for source, _ in requests
            if isinstance(source, BatchDataSource)
        }
        if len(keys) != 1:
            raise NotImplementedError(
                f"Type: {cls} have not implemented how to load fact data with multiple sources."
            )

        client = databricks_fe.FeatureEngineeringClient()

        result_request: RetrievalRequest | None = None
        lookups = []

        for source, request in requests:
            lookups.append(
                databricks_fe.FeatureLookup(
                    source.table.identifier(),
                    lookup_key=list(request.entity_names),
                    feature_names=request.feature_names,
                    timestamp_lookup_key=request.event_timestamp.name
                    if request.event_timestamp
                    else None,
                )
            )

            if result_request is None:
                result_request = request
            else:
                result_request = result_request.unsafe_combine([request])

        assert lookups, "Found no lookups"
        assert result_request, "A `request_result` was supposed to be created."

        async def load() -> pl.LazyFrame:
            import pyspark.pandas as ps

            df = await facts.to_pandas()

            dataset = client.create_training_set(
                df=ps.from_pandas(df),  # type: ignore
                feature_lookups=lookups,
                label=None,
                exclude_columns=None,
            )

            return pl.from_pandas(dataset.load_df().toPandas()).lazy()

        return RetrievalJob.from_lazy_function(load, result_request)

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        return type(self).multi_source_features_for(facts, [(self, request)])

    async def schema(self) -> dict[str, FeatureType]:
        """Returns the schema for the data source

        ```python
        source = FileSource.parquet_at('test_data/titanic.parquet')
        schema = await source.schema()
        >>> {'passenger_id': FeatureType(name='int64'), ...}
        ```

        Returns:
            dict[str, FeatureType]: A dictionary containing the column name and the feature type
        """
        raise NotImplementedError(f"`schema()` is not implemented for {type(self)}.")

    async def freshness(self, feature: Feature) -> datetime | None:
        """
        my_table_freshenss = await (PostgreSQLConfig("DB_URL")
            .table("my_table")
            .freshness()
        )
        """
        spark = self.config.connection()
        return (
            spark.sql(
                f"SELECT MAX({feature.name}) as {feature.name} FROM {self.table.identifier()}"
            )
            .toPandas()[feature.name]
            .to_list()[0]
        )

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        raise NotImplementedError(type(self))

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        raise NotImplementedError(type(self))

    async def overwrite(
        self,
        job: RetrievalJob,
        request: RetrievalRequest,
        predicate: Expression | None = None,
    ) -> None:
        client = databricks_fe.FeatureEngineeringClient()

        conn = self.config.connection()
        df = conn.createDataFrame(await job.unique_entities().to_pandas())

        client.create_table(
            name=self.table.identifier(), primary_keys=list(request.entity_names), df=df
        )

    def with_config(self, config: DatabricksConnectionConfig) -> UCFeatureTableSource:
        return UCFeatureTableSource(config, self.table)


def features_to_read(
    request: RetrievalRequest, schema: StructType, renamer: Renamer | None = None
) -> list[str]:
    stored_fields = schema.fieldNames()

    logger.debug(f"Stored columns in the source: {stored_fields}")

    columns = list(request.entity_names)

    if not renamer:
        renamer = Renamer.noop()

    invers_renamer = renamer.inverse()
    derived_features = [feat.name for feat in request.derived_features]

    missing_features = []

    for feat in request.all_returned_features:
        db_name = invers_renamer.rename(feat.name)

        if db_name in stored_fields:
            if db_name not in columns:
                columns.append(db_name)
        elif feat.name in derived_features:
            continue
        elif not feat.default_value:
            missing_features.append(feat.name)

    if missing_features:
        raise ValueError(
            f"Missing column(s) {missing_features}. Either add it to the table {request.location}, or add a default value"
            f"Available schema is {schema}"
        )

    if request.event_timestamp:
        db_name = invers_renamer.rename(request.event_timestamp.name)
        if db_name not in columns:
            columns.append(request.event_timestamp.name)

    return columns


def validate_pyspark_schema(
    old: StructType, new: StructType, sub_property: str = ""
) -> None:
    from pyspark.sql.types import ArrayType, StructType

    old_schema = {field.name: field.dataType for field in old.fields}
    new_schema = {field.name: field.dataType for field in new.fields}

    missing_fields = []
    incorrect_schema = []

    for name, dtype in old_schema.items():
        if name not in new_schema:
            missing_fields.append((name, dtype))
        elif dtype != new_schema[name]:
            new_type = new_schema[name]
            if isinstance(dtype, StructType) and isinstance(new_type, StructType):
                validate_pyspark_schema(
                    dtype, new_type, f"{sub_property}.{name}." if sub_property else name
                )
            elif (
                isinstance(dtype, ArrayType)
                and isinstance(new_type, ArrayType)
                and isinstance(dtype.elementType, StructType)
                and isinstance(new_type.elementType, StructType)
            ):
                validate_pyspark_schema(
                    dtype.elementType,
                    new_type.elementType,
                    f"{sub_property}.{name}." if sub_property else name,
                )
            else:
                incorrect_schema.append((name, dtype, new_schema[name]))

    error_message = ""

    if missing_fields:
        error_message += "\n".join(
            [
                f"Missing column '{sub_property}{name}' with data type: {dtype}"
                for name, dtype in missing_fields
            ]
        )

    if incorrect_schema:
        error_message += "\n".join(
            [
                f"Incorrect schema for '{sub_property}{name}' got {new_dtype}, but expected {old_dtype}"
                for name, old_dtype, new_dtype in incorrect_schema
            ]
        )

    if error_message:
        raise ValueError(error_message)


@dataclass
class UnityCatalogTableAllJob(RetrievalJob, DatabricksSource):
    config: DatabricksConnectionConfig
    table: UnityCatalogTableConfig
    request: RetrievalRequest
    _limit: int | None
    where: Expression | None = field(default=None)
    renamer: Renamer | None = field(default=None)

    @property
    def request_result(self) -> RequestResult:
        return self.request.request_result

    @property
    def retrieval_requests(self) -> list[RetrievalRequest]:
        return [self.request]

    def filter(self, condition: FilterRepresentable) -> RetrievalJob:
        if isinstance(condition, str):
            new_where = Expression(column=condition)
        else:
            try:
                new_where = Expression.from_value(condition)
            except Exception:
                return RetrievalJob.filter(self, condition)

        if self.where:
            self.where = self.where & new_where
        else:
            self.where = new_where

        return self

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_spark()).toPandas()

    async def to_spark(self, session: SparkSession | None = None) -> SparkFrame:
        con = session or self.config.connection()
        spark_df = con.read.table(self.table.identifier())

        if self.request.features_to_include:
            cols = features_to_read(self.request, spark_df.schema, self.renamer)
            spark_df = spark_df.select(cols)

            if self.renamer and cols:
                renames = {col: self.renamer.rename(col) for col in cols}
                logger.debug(f"Renaming with map {renames}")
                spark_df = spark_df.withColumnsRenamed(renames)
            else:
                logger.debug(f"Selecting '{cols}'")

        if self.where:
            spark_exp = self.where.to_spark()
            assert spark_exp is not None
            spark_df = spark_df.filter(spark_exp)

        if self._limit:
            spark_df = spark_df.limit(self._limit)

        return spark_df

    async def to_lazy_polars(self) -> pl.LazyFrame:
        try:
            from polars import Catalog

            creds = self.config.storage_provider()

            assert creds is not None
            assert self.config.token is not None

            catalog = Catalog(
                workspace_url=self.config.host.read(),
                bearer_token=self.config.token.read(),
            )
            logger.info(
                f"Trying to use polars as the processing engine when loading {self.table.table.read()}."
            )
            df = catalog.scan_table(
                catalog_name=self.table.catalog.read(),
                namespace=self.table.schema.read(),
                table_name=self.table.table.read(),
                credential_provider=creds,
            )

            if self.where:
                polars_exp = self.where.to_polars()
                assert (
                    polars_exp is not None
                ), f"Unable to transform where to polars exp. This is an internal error so please let the maintainers know by setting up an issue. '{self.where}'"
                df = df.filter(polars_exp)

            if self._limit:
                df = df.limit(self._limit)

            return df

        except Exception:
            logger.info(
                "Missing configuration to use polars as the processing engine.",
                " Will create a spark connection.",
            )
            return pl.from_pandas(
                await self.to_pandas(),
                schema_overrides={
                    feat.name: feat.dtype.polars_type
                    for feat in self.retrieval_requests[0].features
                    if feat.dtype != FeatureType.json()
                },
            ).lazy()


@dataclass
class UCTableSource(CodableBatchDataSource, WritableFeatureSource, DatabricksSource):
    """
    A source that connects to a Databricks Unity Catalog table
    """

    config: DatabricksConnectionConfig
    table: UnityCatalogTableConfig
    should_overwrite_schema: bool = False
    renamer: Renamer | None = None

    type_name = "uc_table"

    def job_group_key(self) -> str:
        return f"uc_table-{self.table.identifier()}"

    def overwrite_schema(self, should_overwrite_schema: bool = True) -> UCTableSource:
        return UCTableSource(
            config=self.config,
            table=self.table,
            should_overwrite_schema=should_overwrite_schema,
            renamer=self.renamer,
        )

    def with_renames(self, renames: dict[str, str] | Renamer | None) -> UCTableSource:
        if isinstance(renames, dict):
            renames = Renamer.noop(renames)

        return UCTableSource(
            self.config, self.table, self.should_overwrite_schema, renames
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return UnityCatalogTableAllJob(
            self.config, self.table, request, limit, renamer=self.renamer
        )

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        job = self.all_data(request, None)

        assert request.event_timestamp_request, "Need an event timestamp to filter on."
        return job.filter(
            pl.col(request.event_timestamp_request.event_timestamp.name).is_between(
                start_date, end_date
            )
        )

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls: type[UCTableSource],
        facts: RetrievalJob,
        requests: list[tuple[UCTableSource, RetrievalRequest]],  # type: ignore
    ) -> RetrievalJob:
        from aligned.sources.local import DateFormatter

        if len(requests) != 1:
            raise NotImplementedError(
                f"Type: {cls} have not implemented how to load fact data with multiple sources."
            )

        source, request = requests[0]

        return FileFactualJob(
            source=source.all_data(request, limit=None),
            date_formatter=DateFormatter.noop(),
            requests=[request],
            facts=facts,
        )

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        return type(self).multi_source_features_for(facts, [(self, request)])

    async def schema(self) -> dict[str, FeatureType]:
        """Returns the schema for the data source

        ```python
        source = FileSource.parquet_at('test_data/titanic.parquet')
        schema = await source.schema()
        >>> {'passenger_id': FeatureType(name='int64'), ...}
        ```

        Returns:
            dict[str, FeatureType]: A dictionary containing the column name and the feature type
        """
        spark = self.config.connection()
        schema = spark.table(self.table.identifier()).schema

        aligned_schema: dict[str, FeatureType] = {}

        for column in schema.fields:
            aligned_schema[column.name] = convert_pyspark_type(column.dataType).dtype
        return aligned_schema

    async def freshness(self, feature: Feature) -> datetime | None:
        """
        my_table_freshenss = await (PostgreSQLConfig("DB_URL")
            .table("my_table")
            .freshness()
        )
        """
        spark = self.config.connection()
        return (
            spark.sql(
                f"SELECT MAX({feature.name}) as {feature.name} FROM {self.table.identifier()}"
            )
            .toPandas()[feature.name]
            .to_list()[0]
        )

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        import pyspark.sql.functions as F

        expected_schema = request.spark_schema()

        conn = self.config.connection()
        spark_df = await job.to_spark(conn)

        df = spark_df.select(
            [
                F.col(field.name).cast(field.dataType).alias(field.name)
                for field in expected_schema.fields
            ]
        )

        if conn.catalog.tableExists(self.table.identifier()):
            schema = conn.table(self.table.identifier()).schema
            validate_pyspark_schema(old=schema, new=df.schema)

        df.write.mode("append").saveAsTable(self.table.identifier())

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        import pyspark.sql.functions as F

        expected_schema = request.spark_schema()

        conn = self.config.connection()
        spark_df = await job.to_spark(conn)

        df = spark_df.select(
            [
                F.col(field.name).cast(field.dataType).alias(field.name)
                for field in expected_schema.fields
            ]
        )

        target_table = self.table.identifier()

        if not conn.catalog.tableExists(target_table):
            await self.insert(job, request)
        else:
            entities = request.entity_names
            on_statement = " AND ".join(
                [f"target.{ent} = source.{ent}" for ent in entities]
            )
            schema = conn.table(target_table).schema
            validate_pyspark_schema(old=schema, new=df.schema)

            temp_table = "new_values"
            df.createOrReplaceTempView(temp_table)
            conn.sql(f"""MERGE INTO {target_table} AS target
USING {temp_table} AS source
ON {on_statement}
WHEN MATCHED THEN
  UPDATE SET *
WHEN NOT MATCHED THEN
  INSERT *""")

    async def overwrite(
        self,
        job: RetrievalJob,
        request: RetrievalRequest,
        predicate: Expression | None = None,
    ) -> None:
        import pyspark.sql.functions as F

        expected_schema = request.spark_schema()

        conn = self.config.connection()
        spark_df = await job.to_spark(conn)

        df = spark_df.select(
            [
                F.col(field.name).cast(field.dataType).alias(field.name)
                for field in expected_schema.fields
            ]
        )
        raise_on_invalid_pyspark_schema(df.schema)
        df.write.mode("overwrite").option(
            "overwriteSchema", self.should_overwrite_schema
        ).saveAsTable(self.table.identifier())

    def with_config(self, config: DatabricksConnectionConfig) -> UCTableSource:
        return UCTableSource(config, self.table)

    async def feature_view_code(self, view_name: str) -> str:
        from aligned.sources.renamer import snake_to_pascal
        from pyspark.sql.types import DataType

        con = self.config.connection()
        columns = con.sql(f"DESCRIBE TABLE {self.table.identifier()}").toPandas()

        source = f"{self.config}"

        if isinstance(self.table.catalog, LiteralValue):
            source += f".catalog('{self.table.catalog.read()}')"
        else:
            source += f".catalog({self.table.catalog})"

        if isinstance(self.table.schema, LiteralValue):
            source += f".schema('{self.table.schema.read()}')"
        else:
            source += f".schema({self.table.schema})"

        if isinstance(self.table.table, LiteralValue):
            source += f".table('{self.table.table.read()}')"
        else:
            source += f".table({self.table.table})"

        uppercased_name = snake_to_pascal(view_name)

        data_types: set[str] = set()
        feature_code = ""
        for row in columns.sort_values("col_name").to_dict(orient="records"):
            name = row["col_name"]
            comment = row["comment"]

            spark_type = convert_pyspark_type(DataType.fromDDL(row["data_type"]))
            dtype = spark_type.dtype.feature_factory

            type_name = dtype.__class__.__name__
            data_types.add(type_name)
            feature_code += f"{name} = {type_name}()"

            for constraint in spark_type.constraints:
                if isinstance(constraint, MaxLength):
                    feature_code += f".max_length({constraint.value})"
                elif isinstance(constraint, MinLength):
                    feature_code += f".min_length({constraint.value})"

            if comment:
                formatted_comment = comment.replace("\n", "\\n")
                feature_code += f'\n    "{formatted_comment}"\n'

            feature_code += "\n    "

        all_types = ", ".join(data_types)

        return f"""from aligned import feature_view, {all_types}
from data_contracts.unity_catalog import DatabricksConnectionConfig

@feature_view(
    name="{view_name}",
    description="A databricks table containing {view_name}",
    source={source},
    tags=['code-generated', 'databricks']
)
class {uppercased_name}:
    {feature_code}"""
