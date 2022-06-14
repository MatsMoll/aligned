from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.job_factory import JobFactory
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest

if TYPE_CHECKING:
    from aladdin.retrival_job import RetrivalJob


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

    job_factories: dict[str, JobFactory]
    sources: dict[str, BatchDataSource]

    def features_for(self, facts: dict[str, list], request: FeatureRequest) -> RetrivalJob:
        from aladdin.retrival_job import CombineFactualJob

        core_requests = {
            self.sources[request.feature_view_name]: request
            for request in request.needed_requests
            if request.feature_view_name in self.sources
        }
        job_factories = {self.job_factories[source.type_name] for source in core_requests.keys()}
        return CombineFactualJob(
            jobs=[factory.facts(facts=facts, sources=core_requests) for factory in job_factories],
            combined_requests=[
                request
                for request in request.needed_requests
                if request.feature_view_name not in self.sources
            ],
        )

    def all_for(self, request: FeatureRequest, limit: int | None = None) -> RetrivalJob:
        source = self.sources[request.name]
        if len(request.needed_requests) != 1:
            raise ValueError("Can't use all_for with a request that has subrequests")
        return self.job_factories[source.type_name].all_data(source, request.needed_requests[0], limit)

    def all_between(self, start_date: datetime, end_date: datetime, request: FeatureRequest) -> RetrivalJob:
        source = self.sources[request.name]
        if len(request.needed_requests) != 1:
            raise ValueError("Can't use all_for with a request that has subrequests")
        return self.job_factories[source.type_name].all_between_dates(
            source, request.needed_requests[0], start_date, end_date
        )
