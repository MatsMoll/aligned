from aligned.request.retrieval_request import RetrievalRequest
from aligned.retrieval_job import RetrievalJob


class VectorIndex:
    def nearest_n_to(
        self, data: RetrievalJob, number_of_records: int, request: RetrievalRequest
    ) -> RetrievalJob:
        raise NotImplementedError(type(self))

    def vector_index_name(self) -> str | None:
        raise NotImplementedError(type(self))
