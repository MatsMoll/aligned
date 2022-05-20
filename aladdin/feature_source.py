from dataclasses import dataclass
from aladdin.data_source.batch_data_source import BatchDataSource

from aladdin.job_factory import JobFactory
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import CombineFactualJob, RetrivalJob

class FeatureSource:

    def features_for(self, facts: dict[str, list], requests: list[RetrivalRequest]) -> RetrivalJob:
        raise NotImplementedError()


@dataclass
class BatchFeatureSource(FeatureSource):

    job_factories: dict[str, JobFactory]
    sources: dict[str, BatchDataSource]
    
    def features_for(self, facts: dict[str, list], needed_requests: list[RetrivalRequest], features: set[str]) -> RetrivalJob:
        core_requests = {self.sources[request.feature_view_name]: request for request in needed_requests if request.feature_view_name in self.sources}
        job_factories = {self.job_factories[source.type_name] for source in core_requests.keys()}
        return CombineFactualJob(
            jobs=[factory.facts(facts=facts, sources=core_requests) for factory in job_factories],
            requested_features=features,
            combined_requests=[request for request in needed_requests if request.feature_view_name not in self.sources]
        )