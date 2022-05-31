from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
from aladdin.data_source.batch_data_source import BatchDataSource

from aladdin.job_factory import JobFactory
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest
from aladdin.retrival_job import FilterJob

if TYPE_CHECKING:
    from aladdin.retrival_job import RetrivalJob

class FeatureSource:

    def features_for(self, facts: dict[str, list], request: FeatureRequest) -> RetrivalJob:
        raise NotImplementedError()

    def all_for(self, request: FeatureRequest, limit: int | None = None) -> RetrivalJob:
        raise NotImplementedError()


@dataclass
class BatchFeatureSource(FeatureSource):

    job_factories: dict[str, JobFactory]
    sources: dict[str, BatchDataSource]
    
    def features_for(self, facts: dict[str, list], request: FeatureRequest) -> RetrivalJob:
        from aladdin.retrival_job import CombineFactualJob
        core_requests = {self.sources[request.feature_view_name]: request for request in request.needed_requests if request.feature_view_name in self.sources}
        job_factories = {self.job_factories[source.type_name] for source in core_requests.keys()}
        return CombineFactualJob(
            jobs=[factory.facts(facts=facts, sources=core_requests) for factory in job_factories],
            combined_requests=[request for request in request.needed_requests if request.feature_view_name not in self.sources]
        )

    def all_for(self, request: FeatureRequest, limit: int | None = None) -> RetrivalJob:
        source = self.sources[request.name]
        if len(request.needed_requests) != 1:
            raise ValueError("Can't use all_for with a request that has subrequests")
        return self.job_factories[source.type_name].all_data(source, request.needed_requests[0], limit)