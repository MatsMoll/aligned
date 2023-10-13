from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import asyncio
import numpy as np
import pandas as pd
import polars as pl

from aligned.data_source.batch_data_source import BatchDataSource
from aligned.request.retrival_request import FeatureRequest, RequestResult, RetrivalRequest
from aligned.retrival_job import RetrivalJob
from aligned.schemas.feature import FeatureLocation, EventTimestamp

if TYPE_CHECKING:
    from datetime import datetime


class FeatureSourceFactory:
    def feature_source(self) -> FeatureSource:
        raise NotImplementedError()


class FeatureSource:
    def features_for(self, facts: RetrivalJob, request: FeatureRequest) -> RetrivalJob:
        raise NotImplementedError()

    async def freshness_for(
        self, locations: dict[FeatureLocation, EventTimestamp]
    ) -> dict[FeatureLocation, datetime]:
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

    def features_for(self, facts: RetrivalJob, request: FeatureRequest) -> RetrivalJob:
        from aligned.retrival_job import CombineFactualJob

        core_requests = [
            (self.sources[request.location.identifier], request)
            for request in request.needed_requests
            if request.location.identifier in self.sources
        ]
        source_groupes = {
            self.sources[request.location.identifier].job_group_key()
            for request in request.needed_requests
            if request.location.identifier in self.sources
        }

        # The combined views basicly, as they have no direct
        combined_requests = [
            request for request in request.needed_requests if request.location.identifier not in self.sources
        ]
        jobs = []
        for source_group in source_groupes:
            requests = [
                (source, req) for source, req in core_requests if source.job_group_key() == source_group
            ]
            has_derived_features = any(req.derived_features for _, req in requests)
            job = (
                self.source_types[source_group]
                .multi_source_features_for(facts=facts, requests=requests)
                .ensure_types([req for _, req in requests])
            )
            if has_derived_features:
                job = job.derive_features()
            jobs.append(job)

        if len(combined_requests) > 0 or len(jobs) > 1:
            return CombineFactualJob(
                jobs=jobs,
                combined_requests=combined_requests,
            ).derive_features()
        else:
            return jobs[0]

    def all_for(self, request: FeatureRequest, limit: int | None = None) -> RetrivalJob:
        if len(request.needed_requests) != 1:
            raise ValueError("Can't use all_for with a request that has subrequests")
        if request.location.identifier not in self.sources:
            raise ValueError(
                (
                    f"Unable to find feature view named '{request.location.identifier}'.",
                    'Make sure it is added to the featuer store',
                )
            )
        return (
            self.sources[request.location.identifier]
            .all_data(request.needed_requests[0], limit)
            .ensure_types(request.needed_requests)
            .derive_features(request.needed_requests)
        )

    def all_between(self, start_date: datetime, end_date: datetime, request: FeatureRequest) -> RetrivalJob:
        if len(request.needed_requests) != 1:
            raise ValueError("Can't use all_for with a request that has subrequests")
        if request.location.identifier not in self.sources:
            raise ValueError(
                (
                    f"Unable to find feature view named '{request.location.identifier}'.",
                    'Make sure it is added to the featuer store',
                )
            )
        return (
            self.sources[request.location.identifier]
            .all_between_dates(request.needed_requests[0], start_date, end_date)
            .ensure_types(request.needed_requests)
            .derive_features(requests=request.needed_requests)
        )

    async def freshness_for(
        self, locations: dict[FeatureLocation, EventTimestamp]
    ) -> dict[FeatureLocation, datetime]:
        locs = list(locations.keys())
        results = await asyncio.gather(
            *[self.sources[loc.identifier].freshness(locations[loc]) for loc in locs]
        )
        return dict(zip(locs, results))


class FactualInMemoryJob(RetrivalJob):
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
    facts: RetrivalJob

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.requests)

    def __init__(self, values: dict[str, Any], requests: list[RetrivalRequest], facts: RetrivalJob) -> None:
        self.values = values
        self.requests = requests
        self.facts = facts

    def key(self, request: RetrivalRequest, entity: str, feature_name: str) -> str:
        return f'{request.location}:{entity}:{feature_name}'

    async def to_pandas(self) -> pd.DataFrame:

        columns = set()
        for request in self.requests:
            for feature in request.all_feature_names:
                columns.add(feature)

        result_df = await self.facts.to_pandas()

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

    async def to_polars(self) -> pl.LazyFrame:
        return pl.from_pandas(await self.to_pandas()).lazy()


@dataclass
class InMemoryFeatureSource(FeatureSource, WritableFeatureSource):

    values: dict[str, Any]

    def features_for(self, facts: RetrivalJob, request: FeatureRequest) -> RetrivalJob:
        return FactualInMemoryJob(self.values, request.needed_requests, facts)

    def key(self, request: RetrivalRequest, entity: str, feature_name: str) -> str:
        return f'{request.location}:{entity}:{feature_name}'

    async def write(self, job: RetrivalJob, requests: list[RetrivalRequest]) -> None:
        data = await job.to_pandas()

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
