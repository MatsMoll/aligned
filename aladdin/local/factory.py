from aladdin.job_factory import JobFactory
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.local.source import FileSource
from aladdin.local.job import LocalFileFactualJob, LocalFileFullJob
from aladdin.retrival_job import FullExtractJob

class LocalFileJobFactory(JobFactory):

    source = FileSource

    def all_data(self, source: FileSource, request: RetrivalRequest, limit: int | None) -> FullExtractJob:
        return LocalFileFullJob(source, request, limit)

    def _facts(self, facts: dict[str, list], requests: dict[FileSource, RetrivalRequest]) -> LocalFileFactualJob:
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