from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from datetime import datetime
from typing import TypeVar

from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import (
    CombineFactualJob,
    DateRangeJob,
    FactualRetrivalJob,
    FullExtractJob,
    RetrivalJob,
)

Source = TypeVar('Source', bound=BatchDataSource)


class JobFactory(ABC):
    @abstractproperty
    def source(self) -> type[Source]:
        pass

    def all_data(self, source: Source, request: RetrivalRequest, limit: int | None) -> FullExtractJob:
        pass

    def all_between_dates(
        self,
        source: Source,
        request: RetrivalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> DateRangeJob:
        pass

    def facts(self, facts: dict[str, list], sources: dict[Source, RetrivalRequest]) -> RetrivalJob:
        grouped_requests: dict[str, dict[Source, RetrivalRequest]] = defaultdict(dict)
        grouped_facts: dict[str, dict[str, list]] = defaultdict(
            lambda: {'event_timestamp': facts['event_timestamp']} if 'event_timestamp' in facts else {}
        )
        for data_source in list(sources.keys()):
            if data_source.type_name != self.source.type_name:
                continue
            request = sources[data_source]
            grouped_requests[data_source.job_group_key()][data_source] = request
            for entity_name in request.entity_names:
                grouped_facts[data_source.job_group_key()][entity_name] = facts[entity_name]

        return CombineFactualJob(
            jobs=[
                self._facts(facts=grouped_facts[job_key], requests=requests)
                for job_key, requests in grouped_requests.items()
            ],
            combined_requests=[],
        )

    @abstractmethod
    def _facts(self, facts: dict[str, list], requests: dict[Source, RetrivalRequest]) -> FactualRetrivalJob:
        pass
