from dataclasses import dataclass, field

from datetime import datetime

import pandas as pd
import polars as pl

from aligned.request.retrival_request import AggregatedFeature, AggregateOver, RetrivalRequest
from aligned.retrival_job import RequestResult, RetrivalJob
from aligned.schemas.feature import Feature
from aligned.sources.local import DataFileReference


class LiteralRetrivalJob(RetrivalJob):

    df: pl.LazyFrame
    requests: list[RetrivalRequest]

    def __init__(self, df: pl.LazyFrame | pd.DataFrame, requests: list[RetrivalRequest]) -> None:
        self.requests = requests
        if isinstance(df, pd.DataFrame):
            self.df = pl.from_pandas(df).lazy()
        else:
            self.df = df

    @property
    def retrival_requests(self) -> list[RetrivalRequest]:
        return self.requests

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    async def to_pandas(self) -> pd.DataFrame:
        return self.df.collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        return self.df


async def aggregate(request: RetrivalRequest, core_data: pl.LazyFrame) -> pl.LazyFrame:

    aggregate_over = request.aggregate_over()

    first_over = list(aggregate_over.keys())[0]
    if len(aggregate_over) == 1 and first_over.window is None:

        exprs = []
        for feat in aggregate_over[first_over]:
            tran = await feat.derived_feature.transformation.transform_polars(core_data, feat.name)

            if not isinstance(tran, pl.Expr):
                raise ValueError(f'Aggregation needs to be an expression, got {tran}')
            exprs.append(tran.alias(feat.name))

        return core_data.groupby(first_over.group_by_names).agg(exprs)

    group_by_names = first_over.group_by_names

    if not first_over.window:
        raise ValueError('Found no time column to aggregate over.')

    time_name = first_over.window.time_column.name

    sorted_data = core_data.sort(time_name)
    results = None

    for over, features in aggregate_over.items():
        exprs = []
        for feat in features:
            tran = await feat.derived_feature.transformation.transform_polars(core_data, feat.name)

            if not isinstance(tran, pl.Expr):
                raise ValueError(f'Aggregation needs to be an expression, got {tran}')
            exprs.append(tran.alias(feat.name))

        if not over.window:
            raise ValueError('No time window spesificed.')

        if over.window.every_interval:
            sub = (
                sorted_data.groupby_dynamic(
                    time_name,
                    every=over.window.every_interval,
                    period=over.window.time_window,
                    by=over.group_by_names,
                    offset=-over.window.time_window,
                )
                .agg(exprs)
                .with_columns(pl.col(time_name) + over.window.time_window)
            ).filter(pl.col(time_name) <= sorted_data.select(pl.col(time_name).max()).collect()[0, 0])
        else:
            sub = sorted_data.groupby_rolling(
                time_name,
                period=over.window.time_window,
                by=over.group_by_names,
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

            results = left_df.join_asof(right_df, on=time_name, by=group_by_names).lazy()
        else:
            results = sub

    if results is None:
        raise ValueError(f'Generated no results for aggregate request {request.name}.')

    return results


@dataclass
class FileFullJob(RetrivalJob):

    source: DataFileReference
    request: RetrivalRequest
    limit: int | None = field(default=None)

    @property
    def request_result(self) -> RequestResult:
        return self.request.request_result

    def file_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable

        entity_names = self.request.entity_names
        all_names = list(self.request.all_required_feature_names.union(entity_names))

        request_features = all_names
        if isinstance(self.source, ColumnFeatureMappable):
            request_features = self.source.feature_identifier_for(all_names)

        columns = dict(zip(request_features, all_names))
        df = df.rename(
            columns=columns,
        )

        if self.limit and df.shape[0] > self.limit:
            return df.iloc[: self.limit]
        else:
            return df

    async def file_transform_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable

        if self.request.aggregated_features:
            first_feature = list(self.request.aggregated_features)[0]
            if first_feature.name in df.columns:
                return df

        entity_names = self.request.entity_names
        all_names = list(self.request.all_required_feature_names.union(entity_names))

        request_features = all_names
        if isinstance(self.source, ColumnFeatureMappable):
            request_features = self.source.feature_identifier_for(all_names)
        renames = {
            org_name: wanted_name
            for org_name, wanted_name in zip(request_features, all_names)
            if org_name != wanted_name
        }
        df = df.rename(mapping=renames)

        if self.request.aggregated_features:
            df = await aggregate(self.request, df)

        if self.limit:
            return df.limit(self.limit)
        else:
            return df

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_polars()).collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        file = await self.source.to_polars()
        return await self.file_transform_polars(file)


@dataclass
class FileDateJob(RetrivalJob):

    source: DataFileReference
    request: RetrivalRequest
    start_date: datetime
    end_date: datetime

    @property
    def request_result(self) -> RequestResult:
        return self.request.request_result

    def file_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable

        entity_names = self.request.entity_names
        all_names = list(self.request.all_required_feature_names.union(entity_names))

        request_features = all_names
        if isinstance(self.source, ColumnFeatureMappable):
            request_features = self.source.feature_identifier_for(all_names)

        df.rename(
            columns=dict(zip(request_features, all_names)),
            inplace=True,
        )

        if self.request.event_timestamp is None:
            raise ValueError(f'Source {self.source} have no event timestamp to filter on')

        event_timestamp_column = self.request.event_timestamp.name
        # Making sure it is in the correct format
        df[event_timestamp_column] = pd.to_datetime(
            df[event_timestamp_column], infer_datetime_format=True, utc=True
        )

        start_date_ts = pd.to_datetime(self.start_date, utc=True)
        end_date_ts = pd.to_datetime(self.end_date, utc=True)
        return df.loc[df[event_timestamp_column].between(start_date_ts, end_date_ts)]

    def file_transform_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable

        entity_names = self.request.entity_names
        all_names = list(self.request.all_required_feature_names.union(entity_names))

        if self.request.event_timestamp is None:
            raise ValueError(f'Source {self.source} have no event timestamp to filter on')

        request_features = all_names
        if isinstance(self.source, ColumnFeatureMappable):
            request_features = self.source.feature_identifier_for(all_names)

        df = df.rename(mapping=dict(zip(request_features, all_names)))
        event_timestamp_column = self.request.event_timestamp.name

        return df.filter(pl.col(event_timestamp_column).is_between(self.start_date, self.end_date))

    async def to_pandas(self) -> pd.DataFrame:
        file = await self.source.read_pandas()
        return self.file_transformations(file)

    async def to_polars(self) -> pl.LazyFrame:
        file = await self.source.to_polars()
        return self.file_transform_polars(file)


async def aggregate_over(
    group: AggregateOver,
    features: set[AggregatedFeature],
    df: pl.LazyFrame,
    event_timestamp_col: str,
    group_by: list[str] | None = None,
) -> pl.LazyFrame:

    if not group_by:
        group_by = ['row_id']

    subset = df
    if group.condition:
        raise NotImplementedError('Condition aggregation not implemented for file data source')

    if group.window:
        event_timestamp = group.window.time_column.name
        end = pl.col(event_timestamp_col)
        start = end - group.window.time_window
        subset = subset.filter(pl.col(event_timestamp).is_between(start, end))

    transformations = []
    for feature in features:
        expr = await feature.derived_feature.transformation.transform_polars(
            subset, feature.derived_feature.name
        )
        if isinstance(expr, pl.Expr):
            transformations.append(expr.alias(feature.name))
        else:
            raise NotImplementedError('Only expressions are supported for file data source')

    return subset.groupby(group_by).agg(transformations)


@dataclass
class FileFactualJob(RetrivalJob):

    source: DataFileReference
    requests: list[RetrivalRequest]
    facts: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    def describe(self) -> str:
        return f'Reading file at {self.source}'

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

        all_features: set[Feature] = set()
        for request in self.requests:
            all_features.update(request.all_required_features)

        result = await self.facts.to_polars()
        event_timestamp_col = 'aligned_event_timestamp'

        event_timestamp_entity_columns = [
            req.event_timestamp_request.entity_column for req in self.requests if req.event_timestamp_request
        ]
        event_timestamp_entity_column = None
        did_rename_event_timestamp = False

        if event_timestamp_entity_columns:
            event_timestamp_entity_column = event_timestamp_entity_columns[0]

            if event_timestamp_entity_column and event_timestamp_entity_column in result:
                result = result.rename({event_timestamp_entity_column: event_timestamp_col})
                did_rename_event_timestamp = True

        row_id_name = 'row_id'
        result = result.with_row_count(row_id_name)

        for request in self.requests:
            entity_names = request.entity_names
            all_names = request.all_required_feature_names.union(entity_names)

            if request.event_timestamp_request:
                using_event_timestamp = event_timestamp_entity_column is not None
            else:
                using_event_timestamp = False

            if request.event_timestamp:
                all_names.add(request.event_timestamp.name)

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

            for entity in request.entities:
                feature_df = feature_df.with_columns(pl.col(entity.name).cast(entity.dtype.polars_type))
                result = result.with_columns(pl.col(entity.name).cast(entity.dtype.polars_type))

            column_selects = list(entity_names.union({'row_id'}))

            if using_event_timestamp:
                column_selects.append(event_timestamp_col)

            # Need to only select the relevent entities and row_id
            # Otherwise will we get a duplicate column error
            # We also need to remove the entities after the row_id is joined
            new_result: pl.LazyFrame = result.select(column_selects).join(
                feature_df, on=list(entity_names), how='left'
            )
            new_result = new_result.select(pl.exclude(list(entity_names)))

            for group, features in request.aggregate_over().items():
                aggregated_df = await aggregate_over(group, features, new_result, event_timestamp_col)
                new_result = new_result.join(aggregated_df, on='row_id', how='left')

            if request.event_timestamp and using_event_timestamp:
                field = request.event_timestamp.name
                ttl = request.event_timestamp.ttl

                if new_result.select(field).dtypes[0] == pl.Utf8():
                    new_result = new_result.with_columns(
                        pl.col(field).str.strptime(pl.Datetime, '%+').alias(field)
                    )
                if ttl:
                    ttl_request = (pl.col(field) <= pl.col(event_timestamp_col)) & (
                        pl.col(field) >= pl.col(event_timestamp_col) - ttl
                    )
                    new_result = new_result.filter(pl.col(field).is_null() | ttl_request)
                else:
                    new_result = new_result.filter(
                        pl.col(field).is_null() | (pl.col(field) <= pl.col(event_timestamp_col))
                    )
                new_result = new_result.sort(field, descending=True).select(pl.exclude(field))
            elif request.event_timestamp:
                new_result = new_result.sort([row_id_name, request.event_timestamp.name], descending=True)

            unique = new_result.unique(subset=row_id_name, keep='first')
            column_selects.remove('row_id')
            result = result.join(unique.select(pl.exclude(column_selects)), on=row_id_name, how='left')
            result = result.select(pl.exclude('.*_right$'))

        if did_rename_event_timestamp:
            result = result.rename({event_timestamp_col: event_timestamp_entity_column})

        return result.select([pl.exclude('row_id')])

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_polars()).collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        return await self.file_transformations(await self.source.to_polars())
