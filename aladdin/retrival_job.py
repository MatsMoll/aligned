from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar
from pandas import DataFrame
import asyncio

from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import FeatureType

from aladdin.request.retrival_request import RetrivalRequest
import pyarrow as pa


class RetrivalJob(ABC):

    @abstractmethod
    async def to_df(self) -> DataFrame:
        pass

    @abstractmethod
    async def to_arrow(self) -> DataFrame:
        pass


Source = TypeVar("Source", bound=BatchDataSource)

class SingleSourceRetrivalJob(RetrivalJob):
    source: Source
    request: RetrivalRequest

    async def compute_derived_featuers(self, df: DataFrame) -> DataFrame:
        for feature_round in self.request.derived_features_order():
            for feature in feature_round:
                df[feature.name] = await feature.transformation.transform(df[feature.depending_on_names])

        return df[list(self.request.all_feature_names.union(self.request.entity_names))]

    async def ensure_types(self, df: DataFrame) -> DataFrame:
        for feature in self.request.all_required_features:
            mask = ~df[feature.name].isnull()
            try:
                df.loc[mask, feature.name] = df.loc[mask, feature.name].str.strip('"')
            except AttributeError as _:
                pass

            if feature.dtype == FeatureType("").datetime:
                import pandas as pd
                df[feature.name] = pd.to_datetime(df[feature.name], infer_datetime_format=True, utc=True)
            elif feature.dtype == FeatureType("").datetime or feature.dtype == FeatureType("").string:
                continue
            else:
                df.loc[mask, feature.name] = df.loc[mask, feature.name].astype(feature.dtype.pandas_type)
        return df

    @abstractmethod
    async def _to_df(self) -> DataFrame:
        pass
        
    async def to_df(self) -> DataFrame:
        df = await self._to_df()
        df = await self.ensure_types(df)
        return await self.compute_derived_featuers(df)

        
class FullExtractJob(SingleSourceRetrivalJob):
    limit: int | None

class DateRangeJob(SingleSourceRetrivalJob):
    start_date: datetime
    end_date: datetime

class FactualRetrivalJob(RetrivalJob):

    requests: set[RetrivalRequest]
    facts: dict[str, list]


    async def compute_derived_featuers(self, df: DataFrame) -> DataFrame:
        all_features = set()
        combined_views: list[DerivedFeature] = []
        for request in self.requests:
            for feature_round in request.derived_features_order():
                for feature in feature_round:
                    if feature.depending_on_views - {request.feature_view_name}:
                        combined_views.append(feature)
                        continue
                    print(df.columns)
                    df[feature.name] = await feature.transformation.transform(df[feature.depending_on_names])
            all_features = all_features.union(request.all_feature_names).union(request.entity_names)

        return df[list(all_features)]

    async def ensure_types(self, df: DataFrame) -> DataFrame:
        for request in self.requests:
            for feature in request.all_required_features:
                mask = ~df[feature.name].isnull()
                try:
                    df.loc[mask, feature.name] = df.loc[mask, feature.name].str.strip('"')
                except AttributeError as _:
                    pass

                if feature.dtype == FeatureType("").datetime:
                    import pandas as pd
                    df[feature.name] = pd.to_datetime(df[feature.name], infer_datetime_format=True, utc=True)
                elif feature.dtype == FeatureType("").datetime or feature.dtype == FeatureType("").string:
                    continue
                else:
                    df.loc[mask, feature.name] = df.loc[mask, feature.name].astype(feature.dtype.pandas_type)
        return df

    async def fill_missing(self, df: DataFrame) -> DataFrame:
        return df

    @abstractmethod
    async def _to_df(self) -> DataFrame:
        pass

    async def to_df(self) -> DataFrame:
        df = await self._to_df()
        df = await self.ensure_types(df)
        df = await self.fill_missing(df)
        return await self.compute_derived_featuers(df)


@dataclass
class CombineFactualJob(RetrivalJob):

    jobs: list[RetrivalJob]
    requested_features: set[str]
    combined_requests: list[RetrivalRequest]

    async def combine_data(self, df: DataFrame) -> DataFrame:
        for request in self.combined_requests:
            for feature in request.derived_features:
                df[feature.name] = await feature.transformation.transform(df[feature.depending_on_names])
        return df


    async def to_df(self) -> DataFrame:
        import pandas as pd
        dfs = await asyncio.gather(*[job.to_df() for job in self.jobs])
        df = pd.concat(dfs, axis=1)
        df = await self.combine_data(df)
        if self.requested_features:
            return df[list(self.requested_features)]
        else:
            return df

    async def to_arrow(self) -> DataFrame:
        return await super().to_arrow()