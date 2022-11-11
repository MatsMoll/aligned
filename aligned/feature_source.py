from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from aligned.data_source.batch_data_source import BatchDataSource
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.retrival_job import FactualRetrivalJob

with suppress(ModuleNotFoundError):
    import dask.dataframe as dd

if TYPE_CHECKING:
    from aligned.retrival_job import RetrivalJob


class FeatureSource:
    def features_for(self, facts: dict[str, list], request: FeatureRequest) -> RetrivalJob:
        raise NotImplementedError()


class WritableFeatureSource:
    async def write(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:
        raise NotImplementedError()


class RangeFeatureSource:
    def all_for(self, request: FeatureRequest, limit: int | None = None) -> RetrivalJob:
        raise NotImplementedError()

    def all_between(self, start_date: datetime, end_date: datetime, request: FeatureRequest) -> RetrivalJob:
        raise NotImplementedError()


@dataclass
class BatchFeatureSource(FeatureSource, RangeFeatureSource):
    """A factory for different type of jobs
    This could either be a "fetch all", or "fetch features for ids"

    This class will then know how to strucutre the query in the correct way
    """

    sources: dict[str, BatchDataSource]

    @property
    def source_types(self) -> dict[str, type[BatchDataSource]]:
        return {source.job_group_key(): type(source) for source in self.sources.values()}

    def features_for(self, facts: dict[str, list], request: FeatureRequest) -> RetrivalJob:
        from aligned.retrival_job import CombineFactualJob

        core_requests = {
            self.sources[request.feature_view_name]: request
            for request in request.needed_requests
            if request.feature_view_name in self.sources
        }
        source_groupes = {
            self.sources[request.feature_view_name].job_group_key()
            for request in request.needed_requests
            if request.feature_view_name in self.sources
        }
        # The combined views basicly, as they have no direct
        requests = [
            request for request in request.needed_requests if request.feature_view_name not in self.sources
        ]
        jobs = [
            self.source_types[source_group].feature_for(
                facts=facts,
                requests={
                    source: req
                    for source, req in core_requests.items()
                    if source.job_group_key() == source_group
                },
            )
            for source_group in source_groupes
        ]
        return CombineFactualJob(
            jobs=jobs,
            combined_requests=requests,
        )

    def all_for(self, request: FeatureRequest, limit: int | None = None) -> RetrivalJob:
        if len(request.needed_requests) != 1:
            raise ValueError("Can't use all_for with a request that has subrequests")
        if request.name not in self.sources:
            raise ValueError(
                (
                    f"Unable to find feature view named '{request.name}'.",
                    'Make sure it is added to the featuer store',
                )
            )
        return self.sources[request.name].all_data(request.needed_requests[0], limit)

    def all_between(self, start_date: datetime, end_date: datetime, request: FeatureRequest) -> RetrivalJob:
        if len(request.needed_requests) != 1:
            raise ValueError("Can't use all_for with a request that has subrequests")
        if request.name not in self.sources:
            raise ValueError(
                (
                    f"Unable to find feature view named '{request.name}'.",
                    'Make sure it is added to the featuer store',
                )
            )
        return self.sources[request.name].all_between_dates(request.needed_requests[0], start_date, end_date)


@dataclass
class FactualInMemoryJob(FactualRetrivalJob):
    """
    A job using a in mem storage, aka a dict.

    This will store the features in the following format:

    values = {
        "feature_view:entity-id:feature-name": value,

        ...
        "titanic_passenger:20:age": 22,
        "titanic_passenger:21:age": 50,
        ...
        "titanic_passenger:20:class": "Eco",
        "titanic_passenger:21:class": "VIP",
    }
    """

    values: dict[str, Any]
    requests: list[RetrivalRequest]
    facts: dict[str, list]

    def key(self, request: RetrivalRequest, entity: str, feature_name: str) -> str:
        return f'{request.feature_view_name}:{entity}:{feature_name}'

    async def _to_df(self) -> pd.DataFrame:

        columns = set()
        for request in self.requests:
            for feature in request.all_feature_names:
                columns.add(feature)

        result_df = pd.DataFrame(self.facts)

        for request in self.requests:
            entity_ids = result_df[list(request.entity_names)]
            entities = entity_ids.sum(axis=1)

            for feature in request.all_feature_names:
                # if feature.name in request.entity_names:
                #     continue
                # Fetch one column at a time
                result_df[feature] = [
                    self.values.get(self.key(request, entity, feature)) for entity in entities
                ]

        return result_df

    async def to_df(self) -> pd.DataFrame:
        return await self._to_df()

    async def _to_dask(self) -> dd.DataFrame:
        return await self._to_df()


@dataclass
class InMemoryFeatureSource(FeatureSource, WritableFeatureSource):

    values: dict[str, Any]

    def features_for(self, facts: dict[str, list], request: FeatureRequest) -> RetrivalJob:
        return FactualInMemoryJob(self.values, request.needed_requests, facts)

    def key(self, request: RetrivalRequest, entity: str, feature_name: str) -> str:
        return f'{request.feature_view_name}:{entity}:{feature_name}'

    async def write(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:
        data = await job.to_df()

        for _, row in data.iterrows():
            # Run one query per row
            for request in requests:
                entity_ids = row[list(request.entity_names)]
                entity_id = ':'.join([str(entity_id) for entity_id in entity_ids])
                if not entity_id:
                    continue

                for feature in request.all_features:
                    feature_key = self.key(request, entity_id, feature.name)
                    value = row[feature.name]
                    if value is not None and not np.isnan(value):
                        self.values[feature_key] = row[feature.name]
