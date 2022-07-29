from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import TypeVar

import pandas as pd

from aladdin.data_source.batch_data_source import ColumnFeatureMappable
from aladdin.feature import Feature
from aladdin.local.source import FileReference, FileSource
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob

try:
    import dask.dataframe as dd
except ModuleNotFoundError:
    import pandas as dd

GenericDataFrame = TypeVar('GenericDataFrame', pd.DataFrame, dd.DataFrame)


@dataclass
class FileFullJob(FullExtractJob):

    source: FileReference
    request: RetrivalRequest
    limit: int | None

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
        content = await self.source.read()
        file = StringIO(str(content, 'utf-8'))
        df = pd.read_csv(file)
        return self.file_transformations(df)

    async def _to_dask(self) -> dd.DataFrame:
        if not isinstance(self.source, FileSource):
            raise ValueError("FileFullJob, can only handle FileSource's as source")

        path = self.source.path
        if path.endswith('.csv'):
            df = dd.read_csv(path)
        else:
            df = dd.read_parquet(path)

        return self.file_transformations(df)


@dataclass
class FileDateJob(DateRangeJob):

    source: FileReference
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
        content = await self.source.read()
        file = StringIO(str(content, 'utf-8'))
        df = pd.read_csv(file)

        return self.file_transformations(df)

    async def _to_dask(self) -> dd.DataFrame:
        if not isinstance(self.source, FileSource):
            raise ValueError("FileFullJob, can only handle FileSource's as source")

        path = self.source.path
        if path.endswith('.csv'):
            df = dd.read_csv(path)
        else:
            df = dd.read_parquet(path)

        return self.file_transformations(df)


@dataclass
class FileFactualJob(FactualRetrivalJob):

    source: FileReference
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
        content = await self.source.read()
        file = StringIO(str(content, 'utf-8'))
        df = pd.read_csv(file)
        return self.file_transformations(df)

    async def _to_dask(self) -> dd.DataFrame:
        if not isinstance(self.source, FileSource):
            raise ValueError("FileFullJob, can only handle FileSource's as source")

        path = self.source.path
        if path.endswith('.csv'):
            df = dd.read_csv(path)
        else:
            df = dd.read_parquet(path)

        return self.file_transformations(df)
