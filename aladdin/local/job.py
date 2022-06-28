from dataclasses import dataclass
from datetime import datetime
from io import StringIO

import pandas as pd

from aladdin.data_source.batch_data_source import ColumnFeatureMappable
from aladdin.feature import Feature
from aladdin.local.source import FileReference
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob


@dataclass
class FileFullJob(FullExtractJob):

    source: FileReference
    request: RetrivalRequest
    limit: int | None

    async def _to_df(self) -> pd.DataFrame:
        content = await self.source.read()
        file = StringIO(str(content, 'utf-8'))
        entity_names = self.request.entity_names
        all_names = list(self.request.all_required_feature_names.union(entity_names))

        request_features = all_names
        if isinstance(self.source, ColumnFeatureMappable):
            request_features = self.source.feature_identifier_for(all_names)

        df = pd.read_csv(file)
        df.rename(
            columns={org_name: wanted_name for org_name, wanted_name in zip(request_features, all_names)},
            inplace=True,
        )

        if self.limit and df.shape[0] > self.limit:
            return df.iloc[: self.limit]
        else:
            return df

    async def to_arrow(self) -> pd.DataFrame:
        return await super().to_arrow()


@dataclass
class FileDateJob(DateRangeJob):

    source: FileReference
    request: RetrivalRequest
    start_date: datetime
    end_date: datetime

    async def _to_df(self) -> pd.DataFrame:
        content = await self.source.read()
        file = StringIO(str(content, 'utf-8'))
        entity_names = self.request.entity_names
        all_names = list(self.request.all_required_feature_names.union(entity_names))

        request_features = all_names
        if isinstance(self.source, ColumnFeatureMappable):
            request_features = self.source.feature_identifier_for(all_names)

        df = pd.read_csv(file)
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

    async def to_arrow(self) -> pd.DataFrame:
        return await super().to_arrow()


@dataclass
class FileFactualJob(FactualRetrivalJob):

    source: FileReference
    requests: list[RetrivalRequest]
    facts: dict[str, list]

    async def _to_df(self) -> pd.DataFrame:
        content = await self.source.read()
        file = StringIO(str(content, 'utf-8'))
        df = pd.read_csv(file)
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
            feature_df.rename(
                columns={org_name: wanted_name for org_name, wanted_name in zip(request_features, all_names)},
                inplace=True,
            )
            result.loc[set_mask, list(all_names)] = feature_df.reset_index(drop=True)

        return result

    async def to_arrow(self) -> pd.DataFrame:
        return await super().to_arrow()
