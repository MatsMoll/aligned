from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import polars as pl

from aligned.local.source import DataFileReference
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob, RequestResult
from aligned.schemas.feature import Feature


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

    def file_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable

        all_features: set[Feature] = set()
        for request in self.requests:
            all_features.update(request.all_required_features)

        result = pd.DataFrame(self.facts)

        for request in self.requests:
            entity_names = request.entity_names
            all_names = request.all_required_feature_names.union(entity_names)

            request_features = all_names
            if isinstance(self.source, ColumnFeatureMappable):
                request_features = self.source.feature_identifier_for(all_names)

            mask = pd.Series.repeat(pd.Series([True]), df.shape[0]).reset_index(drop=True)
            set_mask = pd.Series.repeat(pd.Series([True]), result.shape[0]).reset_index(drop=True)
            for entity in entity_names:
                entity
                if isinstance(self.source, ColumnFeatureMappable):
                    entity_source_name = self.source.feature_identifier_for([entity])[0]

                mask = mask & (df[entity_source_name].isin(self.facts[entity]))

                set_mask = set_mask & (pd.Series(self.facts[entity]).isin(df[entity_source_name]))

            feature_df = df.loc[mask, request_features]
            feature_df = feature_df.rename(
                columns={org_name: wanted_name for org_name, wanted_name in zip(request_features, all_names)},
            )
            result.loc[set_mask, list(all_names)] = feature_df.reset_index(drop=True)

        return result

    async def to_pandas(self) -> pd.DataFrame:
        file = await self.source.read_pandas()
        return self.file_transformations(file)

    async def to_polars(self) -> pl.LazyFrame:
        return pl.from_pandas(await self.to_pandas())
