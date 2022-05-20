from aladdin.job_factory import JobFactory
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.local.source import LocalFileSource
from aladdin.local.job import LocalFileFactualJob

class LocalFileJobFactory(JobFactory):

    source = LocalFileSource

    def _facts(self, facts: dict[str, list], requests: dict[LocalFileSource, RetrivalRequest]) -> LocalFileFactualJob:
        if len(requests.keys()) != 1:
            raise ValueError(f"Only able to load one {self.source} at a time")

        data_source = list(requests.keys())[0]
        if not isinstance(data_source, self.source):
            raise ValueError(f"Only {self.source} is supported, recived: {data_source}")

        # Group based on config
        return LocalFileFactualJob(
            source=data_source,
            requests=list(requests.values()),
            facts=facts,
        )