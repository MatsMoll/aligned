from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeVar

import pandas as pd

from aligned.data_source.batch_data_source import ColumnFeatureMappable
from aligned.local.source import DataFileReference
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob
from aligned.schemas.feature import Feature

try:
    import dask.dataframe as dd
except ModuleNotFoundError:
    import pandas as dd

GenericDataFrame = TypeVar('GenericDataFrame', pd.DataFrame, dd.DataFrame)


@dataclass
class FileFullJob(FullExtractJob):

    source: DataFileReference
    request: RetrivalRequest
    limit: int | None = field(default=None)

    def file_transformations(self, df: GenericDataFrame) -> GenericDataFrame:
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

    async def _to_df(self) -> pd.DataFrame:
        file = await self.source.read_pandas()
        return self.file_transformations(file)

    async def _to_dask(self) -> dd.DataFrame:
        file = await self.source.read_dask()
        return self.file_transformations(file)


@dataclass
class FileDateJob(DateRangeJob):

    source: DataFileReference
    request: RetrivalRequest
    start_date: datetime
    end_date: datetime

    def file_transformations(self, df: GenericDataFrame) -> GenericDataFrame:

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

    async def _to_df(self) -> pd.DataFrame:
        file = await self.source.read_pandas()
        return self.file_transformations(file)

    async def _to_dask(self) -> dd.DataFrame:
        file = await self.source.read_dask()
        return self.file_transformations(file)


@dataclass
class FileFactualJob(FactualRetrivalJob):

    source: DataFileReference
    requests: list[RetrivalRequest]
    facts: dict[str, list]

    def file_transformations(self, df: GenericDataFrame) -> GenericDataFrame:
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

    async def _to_df(self) -> pd.DataFrame:
        file = await self.source.read_pandas()
        return self.file_transformations(file)

    async def _to_dask(self) -> dd.DataFrame:
        file = await self.source.read_dask()
        return self.file_transformations(file)
