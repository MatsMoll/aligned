from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING
from functools import partial

from pyiceberg.exceptions import NoSuchTableError
from aligned.config_value import ConfigValue
from aligned.data_source.batch_data_source import CodableBatchDataSource
from aligned.feature_source import WritableFeatureSource
from aligned.local.job import FileFactualJob
from aligned.request.retrieval_request import RetrievalRequest
from aligned.retrieval_job import RetrievalJob
from aligned.schemas.codable import Codable
from dataclasses import dataclass, field

import polars as pl

from aligned.schemas.transformation import Expression
from aligned.sources.local import Deletable

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog


@dataclass
class IcebergCatalog(Codable):
    name: ConfigValue = field(default=ConfigValue.from_value("default"))
    config: dict[str, ConfigValue] = field(
        default_factory=lambda: {
            "type": ConfigValue.from_value("in-memory"),
            "warehouse": ConfigValue.from_value("/tmp/pyiceberg/warehouse"),
        }
    )

    _catalog: Catalog | None = None

    def catalog(self) -> Catalog:
        from pyiceberg.catalog import load_catalog

        if self._catalog is None:
            self._catalog = load_catalog(
                self.name.read(),
                **{key: value.read() for key, value in self.config.items()},
            )

        return self._catalog

    def table(
        self, name: str | ConfigValue, schema: str | ConfigValue | None = None
    ) -> IcebergTable:
        return IcebergTable(
            self,
            table=ConfigValue.from_value(name),
            _schema=ConfigValue.from_value(schema or "default"),
        )


async def read_iceberg_table(
    table: tuple[str, str], catalog: IcebergCatalog, request: RetrievalRequest
) -> pl.LazyFrame:
    iccat = catalog.catalog()
    try:
        tbl = iccat.load_table(table)
        return tbl.to_polars().select(request.read_columns)
    except NoSuchTableError:
        return pl.DataFrame([], schema=request.polars_schema()).lazy()


@dataclass
class IcebergTable(CodableBatchDataSource, Deletable, WritableFeatureSource):
    catalog: IcebergCatalog

    table: ConfigValue
    _schema: ConfigValue

    type_name: str = "iceberg"

    def identifier(self) -> tuple[str, str]:
        return (self._schema.read(), self.table.read())

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.catalog.name.read()}/{self.table.read()}"

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    async def delete(self, predicate: Expression | None = None) -> None:
        catalog = self.catalog.catalog()
        with suppress(NoSuchTableError):
            if predicate is None:
                catalog.drop_table(self.identifier())
            else:
                exp = predicate.to_glot()
                assert exp is not None
                catalog.load_table(self.identifier()).delete(exp.sql("spark"))

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        self.catalog.catalog().load_table(self.identifier()).append(
            (await job.to_polars()).select(request.all_returned_columns).to_arrow()
        )

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        catalog = self.catalog.catalog()
        table = catalog.load_table(self.identifier())

        df = (await job.to_polars()).select(request.all_returned_columns).to_arrow()
        table.upsert(df, join_cols=list(request.entity_names))

    async def overwrite(
        self,
        job: RetrievalJob,
        request: RetrievalRequest,
        predicate: Expression | None = None,
    ) -> None:
        cat = self.catalog.catalog()
        table_name = self.identifier()

        cat.create_namespace_if_not_exists(table_name[0])

        df = (await job.to_polars()).select(request.all_returned_columns).to_arrow()

        cat.create_table_if_not_exists(table_name, schema=df.schema)
        table = cat.load_table(table_name)

        if predicate is None:
            table.overwrite(df)
        else:
            exp = predicate.to_glot()
            assert exp is not None
            table.overwrite(df, exp.sql("spark"))

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        return RetrievalJob.from_lazy_function(
            partial(
                read_iceberg_table,
                table=self.identifier(),
                catalog=self.catalog,
                request=request,
            ),
            request=request,
        )

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls, facts: RetrievalJob, requests: list[tuple[IcebergTable, RetrievalRequest]]
    ) -> RetrievalJob:
        if len(requests) != 1:
            raise ValueError(f"Only able to load one {requests} at a time")

        source, req = requests[0]
        if not isinstance(source, cls):
            raise ValueError(f"Only {cls} is supported, received: {source}")

        # Group based on config
        return FileFactualJob(
            source=source.all_data(req, limit=None),
            requests=[req],
            facts=facts,
        )
