from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Generic, TypeVar

from pandas import DataFrame

from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import FeatureType
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.split_strategy import SplitDataSet, SplitStrategy

if TYPE_CHECKING:
    from aladdin.local.source import FileReference


logger = logging.getLogger(__name__)


class RetrivalJob(ABC):
    @abstractmethod
    async def to_df(self) -> DataFrame:
        pass

    @abstractmethod
    async def to_arrow(self) -> DataFrame:
        pass

    async def as_dataset(self, file: FileReference) -> DataFrame:
        import io

        df = await self.to_df()
        # FIXME: Should be included into the feast lib, as a conveniance and reduce user error
        for col in df.columns:  # Can't input UUID's, so need to convert all ids to strings
            if '_id' in col:
                df[col] = df[col].astype(str)

        data_bytes = io.BytesIO()
        df.to_parquet(data_bytes)  # write to BytesIO buffer
        data_bytes.seek(0)
        await file.write(data_bytes.getvalue())
        return df

    def split_with(self, strategy: SplitStrategy, target_column: str) -> SplitJob:
        return SplitJob(self, target_column, strategy)


@dataclass
class SplitJob:

    job: RetrivalJob
    target_column: str
    strategy: SplitStrategy

    async def use_pandas(self) -> SplitDataSet[DataFrame]:
        data = await self.job.to_df()
        return self.strategy.split_pandas(data, self.target_column)


Source = TypeVar('Source', bound=BatchDataSource)


class SingleSourceRetrivalJob(RetrivalJob, Generic[Source]):
    source: Source
    request: RetrivalRequest

    async def compute_derived_featuers(self, df: DataFrame) -> DataFrame:
        for feature_round in self.request.derived_features_order():
            for feature in feature_round:
                df[feature.name] = await feature.transformation.transform(df[feature.depending_on_names])

        return df

    async def ensure_types(self, df: DataFrame) -> DataFrame:
        for feature in self.request.all_required_features:
            mask = ~df[feature.name].isnull()

            with suppress(AttributeError):
                df.loc[mask, feature.name] = df.loc[mask, feature.name].str.strip('"')

            if feature.dtype == FeatureType('').datetime:
                import pandas as pd

                df[feature.name] = pd.to_datetime(df[feature.name], infer_datetime_format=True, utc=True)
            elif feature.dtype == FeatureType('').datetime or feature.dtype == FeatureType('').string:
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

    requests: list[RetrivalRequest]
    facts: dict[str, list]

    async def compute_derived_featuers(self, df: DataFrame) -> DataFrame:
        all_features: set[str] = set()
        combined_views: list[DerivedFeature] = []
        for request in self.requests:
            for feature_round in request.derived_features_order():
                for feature in feature_round:
                    if feature.depending_on_views - {request.feature_view_name}:
                        combined_views.append(feature)
                        continue

                    df[feature.name] = await feature.transformation.transform(df[feature.depending_on_names])
            all_features = all_features.union(request.all_feature_names).union(request.entity_names)

        return df

    async def ensure_types(self, df: DataFrame) -> DataFrame:
        for request in self.requests:
            for feature in request.all_required_features:
                mask = ~df[feature.name].isnull()
                with suppress(AttributeError):
                    df.loc[mask, feature.name] = df.loc[mask, feature.name].str.strip('"')

                try:
                    if feature.dtype == FeatureType('').datetime:
                        import pandas as pd

                        df[feature.name] = pd.to_datetime(
                            df[feature.name], infer_datetime_format=True, utc=True
                        )
                    elif feature.dtype == FeatureType('').datetime or feature.dtype == FeatureType('').string:
                        continue
                    else:
                        df.loc[mask, feature.name] = df.loc[mask, feature.name].astype(
                            feature.dtype.pandas_type
                        )
                except ValueError as error:
                    logger.info(f'Unable to ensure type for {feature.name}, error: {error}')
                    continue
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
        return await self.combine_data(df)

    async def to_arrow(self) -> DataFrame:
        return await super().to_arrow()


@dataclass
class FilterJob(RetrivalJob):

    include_features: set[str]
    job: RetrivalJob

    async def to_df(self) -> DataFrame:
        df = await self.job.to_df()
        if self.include_features:
            return df[list(self.include_features)]
        else:
            return df

    async def to_arrow(self) -> DataFrame:
        return await super().to_arrow()
