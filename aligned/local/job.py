from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from pytz import timezone
from datetime import datetime

import polars as pl

from aligned.exceptions import UnableToFindFileException
from aligned.lazy_imports import pandas as pd
from aligned.request.retrieval_request import (
    AggregatedFeature,
    AggregateOver,
    RetrievalRequest,
)
from aligned.retrieval_job import RequestResult, RetrievalJob
from aligned.schemas.date_formatter import DateFormatter
from aligned.schemas.feature import Feature
from aligned.data_file import DataFileReference
import logging

logger = logging.getLogger(__name__)


class LiteralRetrievalJob(RetrievalJob):
    df: pl.LazyFrame
    requests: list[RetrievalRequest]

    def __init__(
        self,
        df: pl.LazyFrame | pd.DataFrame | pl.DataFrame,
        requests: list[RetrievalRequest],
    ) -> None:
        self.requests = requests

        if isinstance(df, pl.DataFrame):
            self.df = df.lazy()
        elif isinstance(df, pl.LazyFrame):
            self.df = df
        elif isinstance(df, pd.DataFrame):
            self.df = pl.from_pandas(df).lazy()
        else:
            raise ValueError(f"Unsupported type {type(df)}")

    @property
    def loaded_columns(self) -> list[str]:
        return self.df.collect_schema().names()

    @property
    def retrieval_requests(self) -> list[RetrievalRequest]:
        return self.requests

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    def describe(self) -> str:
        return (
            f"Using literal data frame with columns {self.df.collect_schema().names()}"
        )

    async def to_pandas(self) -> pd.DataFrame:
        return self.df.collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        return self.df


async def aggregate(request: RetrievalRequest, core_data: pl.LazyFrame) -> pl.LazyFrame:
    from aligned import ContractStore

    aggregate_over = request.aggregate_over()

    first_over = list(aggregate_over.keys())[0]
    if len(aggregate_over) == 1 and first_over.window is None:
        exprs = []
        for feat in aggregate_over[first_over]:
            tran = await feat.derived_feature.transformation.transform_polars(
                core_data, feat.name, ContractStore.empty()
            )

            if not isinstance(tran, pl.Expr):
                raise ValueError(f"Aggregation needs to be an expression, got {tran}")
            exprs.append(tran.alias(feat.name))

        return core_data.group_by(first_over.group_by_names).agg(exprs)

    group_by_names = first_over.group_by_names

    if not first_over.window:
        raise ValueError("Found no time column to aggregate over.")

    time_name = first_over.window.time_column.name

    sorted_data = core_data.sort(time_name)
    results = None

    for over, features in aggregate_over.items():
        exprs = []
        for feat in features:
            tran = await feat.derived_feature.transformation.transform_polars(
                core_data, feat.name, ContractStore.empty()
            )

            if not isinstance(tran, pl.Expr):
                raise ValueError(f"Aggregation needs to be an expression, got {tran}")
            exprs.append(tran.alias(feat.name))

        if not over.window:
            raise ValueError("No time window spesificed.")

        if over.window.every_interval:
            sub = (
                sorted_data.group_by_dynamic(
                    time_name,
                    every=over.window.every_interval,
                    period=over.window.time_window,
                    group_by=over.group_by_names,
                    offset=-over.window.time_window,
                )
                .agg(exprs)
                .with_columns(pl.col(time_name) + over.window.time_window)
            ).filter(
                pl.col(time_name)
                <= sorted_data.select(pl.col(time_name).max()).collect()[0, 0]
            )
        else:
            sub = sorted_data.rolling(
                time_name,
                period=over.window.time_window,
                group_by=over.group_by_names,
            ).agg(exprs)

        if over.window.offset_interval:
            sub = sub.with_columns(pl.col(time_name) - over.window.offset_interval)

        if results is not None:
            existing_result = results.collect()
            new_aggregations = sub.collect()

            if existing_result.shape[0] > new_aggregations.shape[0]:
                left_df = existing_result
                right_df = new_aggregations
            else:
                right_df = existing_result
                left_df = new_aggregations

            results = left_df.join_asof(
                right_df, on=time_name, by=group_by_names
            ).lazy()
        else:
            results = sub

    if results is None:
        raise ValueError(f"Generated no results for aggregate request {request.name}.")

    return results


def decode_timestamps(
    df: pl.LazyFrame, request: RetrievalRequest, formatter: DateFormatter
) -> pl.LazyFrame:
    decode_columns: dict[str, str | None] = {}
    check_timezone_columns: dict[str, str | None] = {}

    dtypes = df.collect_schema()

    all_features = request.all_features

    if request.event_timestamp:
        all_features.add(request.event_timestamp.as_feature())

    for feature in all_features:
        if feature.dtype.is_datetime and feature.name in dtypes:
            if not isinstance(dtypes[feature.name], pl.Datetime):
                decode_columns[feature.name] = feature.dtype.datetime_timezone
            else:
                check_timezone_columns[feature.name] = feature.dtype.datetime_timezone

    exprs = []

    for column, time_zone in decode_columns.items():
        logger.info(
            f"Decoding column {column} using {formatter} with timezone {time_zone}"
        )

        if time_zone is None:
            exprs.append(
                formatter.decode_polars(column).dt.replace_time_zone(None).alias(column)
            )
        else:
            exprs.append(
                formatter.decode_polars(column)
                .dt.convert_time_zone(time_zone)
                .alias(column)
            )

    for column, time_zone in check_timezone_columns.items():
        logger.info(f"Checking timezone for column {column} with timezone {time_zone}")
        if time_zone is None:
            exprs.append(pl.col(column).dt.replace_time_zone(None).alias(column))
        else:
            exprs.append(pl.col(column).dt.convert_time_zone(time_zone).alias(column))

    return df.with_columns(exprs)


@dataclass
class FileFullJob(RetrievalJob):
    source: DataFileReference | RetrievalJob
    request: RetrievalRequest
    limit: int | None = field(default=None)
    date_formatter: DateFormatter = field(default_factory=DateFormatter.iso_8601)

    @property
    def loaded_columns(self) -> list[str]:
        if isinstance(self.source, DataFileReference):
            return []
        else:
            return self.source.loaded_columns

    @property
    def request_result(self) -> RequestResult:
        return self.request.request_result

    @property
    def retrieval_requests(self) -> list[RetrievalRequest]:
        return [self.request]

    def describe(self) -> str:
        return f"Reading everything form file {self.source}."

    async def file_transform_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable
        from aligned.sources.local import fill_missing_in_request

        if not self.request.features_to_include:
            return df

        if self.request.aggregated_features:
            first_feature = list(self.request.aggregated_features)[0]
            if first_feature.name in df.collect_schema().names():
                return df

        entity_names = self.request.entity_names
        all_names = list(self.request.all_required_feature_names.union(entity_names))

        request_features = all_names

        feature_column_map = {}
        if isinstance(self.source, ColumnFeatureMappable):
            request_features = self.source.feature_identifier_for(all_names)
            feature_column_map = dict(zip(all_names, request_features))

        df = fill_missing_in_request(self.request, df, feature_column_map)
        renames = {
            org_name: wanted_name
            for org_name, wanted_name in zip(request_features, all_names)
            if org_name != wanted_name
        }
        if renames:
            df = df.rename(mapping=renames)

        df = decode_timestamps(df, self.request, self.date_formatter)

        if self.request.aggregated_features:
            df = await aggregate(self.request, df)

        if self.limit:
            return df.limit(self.limit)
        else:
            return df

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        file = await self.source.to_lazy_polars()
        return await self.file_transform_polars(file)


@dataclass
class FileDateJob(RetrievalJob):
    source: DataFileReference
    request: RetrievalRequest
    start_date: datetime
    end_date: datetime
    date_formatter: DateFormatter = field(default_factory=DateFormatter.iso_8601)

    @property
    def request_result(self) -> RequestResult:
        return self.request.request_result

    @property
    def retrieval_requests(self) -> list[RetrievalRequest]:
        return [self.request]

    def file_transform_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable
        from aligned.sources.local import fill_missing_in_request

        if not self.request.features_to_include:
            return df

        entity_names = self.request.entity_names
        all_names = list(self.request.all_required_feature_names.union(entity_names))

        if self.request.event_timestamp is None:
            raise ValueError(
                f"Source {self.source} have no event timestamp to filter on"
            )

        request_features = all_names
        feature_column_map = {}
        if isinstance(self.source, ColumnFeatureMappable):
            request_features = self.source.feature_identifier_for(all_names)
            feature_column_map = dict(zip(all_names, request_features))

        df = fill_missing_in_request(self.request, df, feature_column_map)

        df = df.rename(mapping=dict(zip(request_features, all_names)))
        event_timestamp_column = self.request.event_timestamp.name
        df = decode_timestamps(df, self.request, self.date_formatter)

        time_zone = self.request.event_timestamp.dtype.datetime_timezone
        if time_zone is None:
            start_date = self.start_date.replace(tzinfo=None)
            end_date = self.end_date.replace(tzinfo=None)
        else:
            tz = timezone(time_zone)
            if self.start_date == datetime.min:
                start_date = self.start_date.replace(tzinfo=tz)
            else:
                start_date = self.start_date.astimezone(tz)

            if self.end_date == datetime.max:
                end_date = self.end_date.replace(tzinfo=tz)
            else:
                end_date = self.end_date.astimezone(tz)

        return df.filter(
            pl.col(event_timestamp_column).is_between(start_date, end_date)
        )

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        file = await self.source.to_lazy_polars()
        return self.file_transform_polars(file)


async def aggregate_over(
    group: AggregateOver,
    features: set[AggregatedFeature],
    df: pl.LazyFrame,
    event_timestamp_col: str,
    group_by: list[str] | None = None,
) -> pl.LazyFrame:
    from aligned import ContractStore

    if not group_by:
        group_by = ["row_id"]

    subset = df
    if group.condition:
        raise NotImplementedError(
            "Condition aggregation not implemented for file data source"
        )

    if group.window:
        event_timestamp = group.window.time_column.name
        end = pl.col(event_timestamp_col)
        start = end - group.window.time_window
        subset = subset.filter(pl.col(event_timestamp).is_between(start, end))

    transformations = []
    for feature in features:
        expr = await feature.derived_feature.transformation.transform_polars(
            subset, feature.derived_feature.name, ContractStore.empty()
        )
        if isinstance(expr, pl.Expr):
            transformations.append(expr.alias(feature.name))
        else:
            raise NotImplementedError(
                "Only expressions are supported for file data source"
            )

    return subset.group_by(group_by).agg(transformations)


@dataclass
class FileFactualJob(RetrievalJob):
    source: DataFileReference | RetrievalJob
    requests: list[RetrievalRequest]
    facts: RetrievalJob
    date_formatter: DateFormatter = field(default_factory=DateFormatter.iso_8601)

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    @property
    def retrieval_requests(self) -> list[RetrievalRequest]:
        return self.requests

    def describe(self) -> str:
        return f"Reading file at {self.source}"

    async def file_transformations(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Selects only the wanted subset from the loaded source

        ```python
        await self.file_transformations(await self.source.to_polars())
        ```

        Args:
            df (pl.LazyFrame): The loaded file source which contains all features

        Returns:
            pl.LazyFrame: The subset of the source which is needed for the request
        """
        from aligned.data_source.batch_data_source import ColumnFeatureMappable
        from aligned.sources.local import fill_missing_in_request

        all_features: set[Feature] = set()
        date_features: set[str] = set()

        for request in self.requests:
            all_features.update(request.all_required_features)
            if request.event_timestamp:
                date_features.add(request.event_timestamp.name)
            for feature in request.features:
                if feature.dtype.is_datetime:
                    date_features.add(feature.name)

        result = await self.facts.to_lazy_polars()
        event_timestamp_col = "aligned_event_timestamp"

        event_timestamp_entity_columns = [
            req.event_timestamp_request.entity_column
            for req in self.requests
            if req.event_timestamp_request
        ]
        event_timestamp_entity_column = None
        did_rename_event_timestamp = False

        if event_timestamp_entity_columns:
            event_timestamp_entity_column = event_timestamp_entity_columns[0]

            if (
                event_timestamp_entity_column
                and event_timestamp_entity_column in result
            ):
                result = result.rename(
                    {event_timestamp_entity_column: event_timestamp_col}
                )
                did_rename_event_timestamp = True

        row_id_name = "row_id"
        result = result.with_row_index(row_id_name)

        for request in self.requests:
            entity_names = request.entity_names
            assert entity_names, "Need at there will be at least one entity to join"
            all_names = request.all_required_feature_names.union(entity_names)

            request_features = list(all_names)
            feature_column_map = {}
            if isinstance(self.source, ColumnFeatureMappable):
                request_features = self.source.feature_identifier_for(list(all_names))
                feature_column_map = dict(zip(all_names, request_features))

            df = fill_missing_in_request(request, df, feature_column_map)
            df_columns = df.collect_schema().names()

            for derived_feature in request.derived_features:
                if derived_feature.name in df_columns:
                    all_names.add(derived_feature.name)

            column_selects = list(entity_names.union({"row_id"}))

            if request.event_timestamp_request:
                using_event_timestamp = event_timestamp_entity_column is not None
            else:
                using_event_timestamp = False

            if request.event_timestamp:
                all_names.add(request.event_timestamp.name)

            if using_event_timestamp:
                column_selects.append(event_timestamp_col)

            missing_agg_features = [
                feat
                for feat in request.aggregated_features
                if feat.name not in df_columns
            ]
            if request.aggregated_features and not missing_agg_features:
                new_result = result.join(
                    df.select(request.all_returned_columns),
                    on=list(entity_names),
                    how="left",
                )
            else:
                all_names = list(all_names)

                request_features = list(all_names)
                if isinstance(self.source, ColumnFeatureMappable):
                    request_features = self.source.feature_identifier_for(all_names)

                feature_df = df.select(request_features)

                renames = {
                    org_name: wanted_name
                    for org_name, wanted_name in zip(request_features, all_names)
                    if org_name != wanted_name
                }
                if renames:
                    feature_df = feature_df.rename(renames)

                feature_df = decode_timestamps(feature_df, request, self.date_formatter)

                for entity in request.entities:
                    feature_df = feature_df.with_columns(
                        pl.col(entity.name).cast(entity.dtype.polars_type)
                    )
                    result = result.with_columns(
                        pl.col(entity.name).cast(entity.dtype.polars_type)
                    )

                # Need to only select the relevant entities and row_id
                # Otherwise will we get a duplicate column error
                # We also need to remove the entities after the row_id is joined
                new_result: pl.LazyFrame = result.select(column_selects).join(
                    feature_df, on=list(entity_names), how="left"
                )
                new_result = new_result.select(pl.exclude(list(entity_names)))

                for group, features in request.aggregate_over().items():
                    missing_features = [
                        feature.name
                        for feature in features
                        if feature.name not in df.collect_schema().names()
                    ]
                    if missing_features:
                        aggregated_df = await aggregate_over(
                            group, features, new_result, event_timestamp_col
                        )
                        new_result = new_result.join(
                            aggregated_df, on="row_id", how="left"
                        )

            if request.event_timestamp and using_event_timestamp:
                field = request.event_timestamp.name
                ttl = request.event_timestamp.ttl

                if ttl:
                    ttl_request = (pl.col(field) <= pl.col(event_timestamp_col)) & (
                        pl.col(field) >= pl.col(event_timestamp_col) - ttl
                    )
                    new_result = new_result.filter(
                        pl.col(field).is_null() | ttl_request
                    )
                else:
                    new_result = new_result.filter(
                        pl.col(field).is_null()
                        | (pl.col(field) <= pl.col(event_timestamp_col))
                    )
                new_result = new_result.sort(
                    field, descending=True, nulls_last=True
                ).select(pl.exclude(field))
            elif request.event_timestamp:
                new_result = new_result.sort(
                    [row_id_name, request.event_timestamp.name], descending=True
                )
                if request.event_timestamp.name not in request.features_to_include:
                    new_result = new_result.select(
                        pl.exclude(request.event_timestamp.name)
                    )

            unique = new_result.unique(subset=row_id_name, keep="first")
            column_selects.remove("row_id")
            result = result.join(
                unique.select(pl.exclude(column_selects)),
                on=row_id_name,
                how="left",
                coalesce=True,
            )
            result = result.select(pl.exclude(".*_right"))

        if did_rename_event_timestamp:
            result = result.rename({event_timestamp_col: event_timestamp_entity_column})  # type: ignore

        return result.select([pl.exclude("row_id")])

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_lazy_polars()).collect().to_pandas()

    async def to_lazy_polars(self) -> pl.LazyFrame:
        try:
            return await self.file_transformations(await self.source.to_lazy_polars())
        except UnableToFindFileException:
            entities = await self.facts.to_lazy_polars()

            columns: list[pl.Expr] = []
            for req in self.retrieval_requests:
                columns.extend(
                    [
                        pl.lit(
                            feat.default_value.python_value
                            if feat.default_value
                            else None
                        ).alias(feat.name)
                        for feat in req.all_returned_features
                    ]
                )

            return entities.with_columns(columns)

    def log_each_job(
        self, logger_func: Callable[[object], None] | None = None
    ) -> RetrievalJob:
        if isinstance(self.source, RetrievalJob):
            return FileFactualJob(
                self.source.log_each_job(logger_func),
                self.requests,
                self.facts.log_each_job(logger_func),
            )
        else:
            return self
