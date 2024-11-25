from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import asyncio

from aligned.data_source.batch_data_source import BatchDataSource
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.retrival_job import RetrivalJob
from aligned.schemas.feature import FeatureLocation, Feature

if TYPE_CHECKING:
    from datetime import datetime


class FeatureSourceFactory:
    def feature_source(self) -> FeatureSource:
        raise NotImplementedError()


class FeatureSource:
    def features_for(self, facts: RetrivalJob, request: FeatureRequest) -> RetrivalJob:
        raise NotImplementedError()

    async def freshness_for(
        self, locations: dict[FeatureLocation, Feature]
    ) -> dict[FeatureLocation, datetime | None]:
        raise NotImplementedError()


class WritableFeatureSource:
    async def insert(self, job: RetrivalJob, request: RetrivalRequest) -> None:
        raise NotImplementedError(f'Append is not implemented for {type(self)}.')

    async def upsert(self, job: RetrivalJob, request: RetrivalRequest) -> None:
        raise NotImplementedError(f'Upsert write is not implemented for {type(self)}.')

    async def overwrite(self, job: RetrivalJob, request: RetrivalRequest) -> None:
        raise NotImplementedError(f'Overwrite write is not implemented for {type(self)}.')


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

    sources: Mapping[str, BatchDataSource]

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

        loaded_columns = set(facts.loaded_columns)

        def needs_to_load_source(requests: list[RetrivalRequest]) -> bool:
            for req in requests:
                if set(req.feature_names) - loaded_columns:
                    return True

                for feat in req.derived_features:
                    if (
                        set(
                            depends_on.name
                            for depends_on in feat.depending_on
                            if depends_on.location != req.location
                        )
                        - loaded_columns
                    ):
                        return True

                for feat in req.aggregated_features:
                    if set(feat.depending_on_names) - loaded_columns:
                        return True
            return False

        # The combined views basicly, as they have no direct
        combined_requests = [
            request for request in request.needed_requests if request.location.identifier not in self.sources
        ]
        jobs = []
        for source_group in source_groupes:
            requests_with_source = [
                (source, req) for source, req in core_requests if source.job_group_key() == source_group
            ]
            requests = [req for _, req in requests_with_source]

            if needs_to_load_source(requests):
                job = (
                    self.source_types[source_group]
                    .multi_source_features_for(facts=facts, requests=requests_with_source)
                    .ensure_types(requests)
                    .derive_features()
                )
            else:
                job = facts.derive_features(requests)

            if len(requests) == 1 and requests_with_source[0][1].aggregated_features:
                req = requests_with_source[0][1]
                job = job.aggregate(req)

            jobs.append(job)

        fact_features = loaded_columns - set(request.request_result.entity_columns)
        if fact_features:
            jobs.append(facts)

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
        self, locations: dict[FeatureLocation, Feature]
    ) -> dict[FeatureLocation, datetime | None]:
        locs = list(locations.keys())
        results = await asyncio.gather(
            *[self.sources[loc.identifier].freshness(locations[loc]) for loc in locs]
        )
        return dict(zip(locs, results))
