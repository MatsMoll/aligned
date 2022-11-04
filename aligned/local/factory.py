from datetime import datetime

from aligned.job_factory import JobFactory
from aligned.local.job import FileDateJob, FileFactualJob, FileFullJob
from aligned.local.source import CsvFileSource, ParquetFileSource
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import DateRangeJob, FactualRetrivalJob, FullExtractJob


class CsvFileJobFactory(JobFactory):

    source = CsvFileSource

    def all_data(self, source: CsvFileSource, request: RetrivalRequest, limit: int | None) -> FullExtractJob:
        return FileFullJob(source, request, limit)

    def all_between_dates(
        self,
        source: CsvFileSource,
        request: RetrivalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> DateRangeJob:
        return FileDateJob(source=source, request=request, start_date=start_date, end_date=end_date)

    def _facts(
        self, facts: dict[str, list], requests: dict[CsvFileSource, RetrivalRequest]
    ) -> FactualRetrivalJob:
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


class ParquetFileJobFactory(JobFactory):

    source = ParquetFileSource

    def all_data(
        self, source: ParquetFileSource, request: RetrivalRequest, limit: int | None
    ) -> FullExtractJob:
        return FileFullJob(source, request, limit)

    def all_between_dates(
        self,
        source: ParquetFileSource,
        request: RetrivalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> DateRangeJob:
        return FileDateJob(source=source, request=request, start_date=start_date, end_date=end_date)

    def _facts(
        self, facts: dict[str, list], requests: dict[ParquetFileSource, RetrivalRequest]
    ) -> FactualRetrivalJob:
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
