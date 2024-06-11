from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import RetrivalJob


class VectorIndex:
    def nearest_n_to(
        self, data: RetrivalJob, number_of_records: int, request: RetrivalRequest
    ) -> RetrivalJob:
        raise NotImplementedError(type(self))

    def vector_index_name(self) -> str | None:
        raise NotImplementedError(type(self))
