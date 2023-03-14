from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import polars as pl

from aligned.local.source import DataFileReference
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob, RequestResult, RetrivalJob
from aligned.schemas.feature import Feature


class LiteralRetrivalJob(RetrivalJob):

    df: pl.LazyFrame
    result: RequestResult

    def __init__(self, df: pl.LazyFrame | pd.DataFrame, result: RequestResult) -> None:
        self.result = result
        if isinstance(df, pd.DataFrame):
            self.df = pl.from_pandas(df).lazy()
        else:
            self.df = df

    @property
    def request_result(self) -> RequestResult:
        return self.result

    async def to_pandas(self) -> pd.DataFrame:
        return self.df.collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        return self.df


@dataclass
class FileFullJob(FullExtractJob):

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

        df = df.rename(
            columns={org_name: wanted_name for org_name, wanted_name in zip(request_features, all_names)},
        )

        if self.limit and df.shape[0] > self.limit:
            return df.iloc[: self.limit]
        else:
            return df

    def file_transform_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable

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

        if self.limit:
            return df.limit(self.limit)
        else:
            return df

    async def to_pandas(self) -> pd.DataFrame:
        file = await self.source.read_pandas()
        return self.file_transformations(file)

    async def to_polars(self) -> pl.LazyFrame:
        file = await self.source.to_polars()
        return self.file_transform_polars(file)


@dataclass
class FileDateJob(DateRangeJob):

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
            columns={org_name: wanted_name for org_name, wanted_name in zip(request_features, all_names)},
            inplace=True,
        )

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

        request_features = all_names
        if isinstance(self.source, ColumnFeatureMappable):
            request_features = self.source.feature_identifier_for(all_names)

        df = df.rename(
            mapping={org_name: wanted_name for org_name, wanted_name in zip(request_features, all_names)}
        )
        event_timestamp_column = self.request.event_timestamp.name

        return df.filter(pl.col(event_timestamp_column).is_between(self.start_date, self.end_date))

    async def to_pandas(self) -> pd.DataFrame:
        file = await self.source.read_pandas()
        return self.file_transformations(file)

    async def to_polars(self) -> pl.LazyFrame:
        file = await self.source.to_polars()
        return self.file_transform_polars(file)


@dataclass
class FileFactualJob(FactualRetrivalJob):

    source: DataFileReference
    requests: list[RetrivalRequest]
    facts: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

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
        event_timestamp_col = 'event_timestamp'
        row_id_name = 'row_id'
        result = result.with_row_count(row_id_name)

        for request in self.requests:
            entity_names = request.entity_names
            all_names = request.all_required_feature_names.union(entity_names)

            if request.event_timestamp:
                all_names.add(request.event_timestamp.name)

            request_features = all_names
            if isinstance(self.source, ColumnFeatureMappable):
                request_features = self.source.feature_identifier_for(all_names)

            feature_df = df.select(request_features)

            renames = {org_name: wanted_name for org_name, wanted_name in zip(request_features, all_names)}
            feature_df = feature_df.rename(renames)

            for entity in request.entities:
                feature_df = feature_df.with_column(pl.col(entity.name).cast(entity.dtype.polars_type))
                result = result.with_column(pl.col(entity.name).cast(entity.dtype.polars_type))

            column_selects = list(entity_names.union({'row_id'}))
            if request.event_timestamp:
                column_selects.append('event_timestamp')

            # Need to only select the relevent entities and row_id
            # Otherwise will we get a duplicate column error
            # We also need to remove the entities after the row_id is joined
            new_result: pl.LazyFrame = result.select(column_selects).join(
                feature_df, on=list(entity_names), how='left'
            )
            new_result = new_result.select(pl.exclude(list(entity_names)))

            if request.event_timestamp:
                new_result = new_result.with_columns(
                    pl.col(request.event_timestamp.name)
                    .str.strptime(pl.Datetime, '%+')
                    .alias(request.event_timestamp.name)
                )
                field = request.event_timestamp.name
                ttl = request.event_timestamp.ttl
                if ttl:
                    ttl_request = (pl.col(field) <= pl.col(event_timestamp_col)) & (
                        pl.col(field) >= pl.col(event_timestamp_col) - ttl
                    )
                    new_result = new_result.filter(pl.col(field).is_null() | ttl_request)
                else:
                    new_result = new_result.filter(
                        pl.col(field).is_null() | (pl.col(field) <= pl.col(event_timestamp_col))
                    )

            unique = new_result.unique(subset=row_id_name, keep='first')
            result = result.join(unique, on=row_id_name, how='left')
            result = result.select(pl.exclude('.*_right$'))

        return result.select([pl.exclude('row_id')])

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_polars()).collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        return await self.file_transformations(await self.source.to_polars())
