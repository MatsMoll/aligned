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
    facts: dict[str, list]

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    def file_transformations(self, df: pl.LazyFrame) -> pl.LazyFrame:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable

        all_features: set[Feature] = set()
        for request in self.requests:
            all_features.update(request.all_required_features)

        result = pl.DataFrame(self.facts).lazy()
        event_timestamp_col = 'event_timestamp'

        for request in self.requests:
            entity_names = request.entity_names
            all_names = request.all_required_feature_names.union(entity_names)

            request_features = all_names
            if isinstance(self.source, ColumnFeatureMappable):
                request_features = self.source.feature_identifier_for(all_names)

            feature_df = df.select(request_features)
            feature_df = feature_df.rename(
                {org_name: wanted_name for org_name, wanted_name in zip(request_features, all_names)},
            )

            for entity in request.entities:
                feature_df = feature_df.with_column(pl.col(entity.name).cast(entity.dtype.polars_type))
                result = result.with_column(pl.col(entity.name).cast(entity.dtype.polars_type))

            new_result: pl.LazyFrame = result.join(feature_df, on=list(entity_names), how='left')

            if request.event_timestamp:
                field = request.event_timestamp.name
                ttl = request.event_timestamp.ttl
                if ttl:
                    ttl_request = (pl.col(field) <= pl.col(event_timestamp_col)) & (
                        pl.col(field) >= pl.col(event_timestamp_col) - ttl
                    )
                    new_result = new_result.filter(pl.col(field).is_null() | ttl_request)
                else:
                    new_result = new_result.filter(
                        pl.col(field).is_null() | pl.col(field) <= pl.col(event_timestamp_col)
                    )

            unique = new_result.unique(subset=list(entity_names), keep='first')

            result = result.join(unique, on=list(entity_names), how='left')

        return result

    async def to_pandas(self) -> pd.DataFrame:
        return (await self.to_polars()).collect().to_pandas()

    async def to_polars(self) -> pl.LazyFrame:
        return self.file_transformations(await self.source.to_polars())
