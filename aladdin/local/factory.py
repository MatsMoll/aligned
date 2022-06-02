from datetime import datetime

from aladdin.job_factory import JobFactory
from aladdin.local.job import FileDateJob, FileFactualJob, FileFullJob
from aladdin.local.source import FileSource
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import RetrivalJob


class LocalFileJobFactory(JobFactory):

    source = FileSource

    def all_data(self, source: FileSource, request: RetrivalRequest, limit: int | None) -> RetrivalJob:
        return FileFullJob(source, request, limit)

    def all_between_dates(
        self,
        source: FileSource,
        request: RetrivalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrivalJob:
        return FileDateJob(source=source, request=request, start_date=start_date, end_date=end_date)

    def _facts(self, facts: dict[str, list], requests: dict[FileSource, RetrivalRequest]) -> RetrivalJob:
        if len(requests.keys()) != 1:
            raise ValueError(f'Only able to load one {self.source} at a time')

        data_source = list(requests.keys())[0]
        if not isinstance(data_source, self.source):
            raise ValueError(f'Only {self.source} is supported, recived: {data_source}')

        # Group based on config
        return FileFactualJob(
            source=data_source,
            requests=list(requests.values()),
            facts=facts,
        )
