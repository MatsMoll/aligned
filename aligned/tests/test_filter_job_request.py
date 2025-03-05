from aligned.retrieval_job import RetrievalJob, RetrievalRequest
from aligned.schemas.feature import Feature, FeatureLocation, FeatureType


def test_filter_job_result_request() -> None:
    job = RetrievalJob.from_dict(
        {"id": [1], "a": [3], "b": ["test"], "c": [10]},
        request=RetrievalRequest(
            name="test",
            location=FeatureLocation.feature_view("test"),
            entities={Feature("id", FeatureType.int32())},
            features={
                Feature("a", FeatureType.int32()),
                Feature("b", FeatureType.string()),
                Feature("c", FeatureType.int32()),
            },
            derived_features=set(),
        ),
    )

    assert set(job.request_result.feature_columns) == {"a", "b", "c"}
    assert set(job.request_result.entity_columns) == {"id"}
    filtered_job = job.select_columns({"b"})
    assert set(filtered_job.request_result.feature_columns) == {"b"}
    assert set(filtered_job.request_result.entity_columns) == {"id"}


def test_filter_job_retrieval_requests() -> None:
    job = RetrievalJob.from_dict(
        {"id": [1], "a": [3], "b": ["test"], "c": [10]},
        request=RetrievalRequest(
            name="test",
            location=FeatureLocation.feature_view("test"),
            entities={Feature("id", FeatureType.int32())},
            features={
                Feature("a", FeatureType.int32()),
                Feature("b", FeatureType.string()),
                Feature("c", FeatureType.int32()),
            },
            derived_features=set(),
        ),
    )

    retrieval_requests = job.retrieval_requests

    assert len(retrieval_requests) == 1
    request = retrieval_requests[0]

    assert request.features_to_include == {"b", "a", "c"}

    assert set(job.request_result.feature_columns) == {"a", "b", "c"}
    assert set(job.request_result.entity_columns) == {"id"}

    filtered_job = job.select_columns({"b"})
    retrieval_requests = filtered_job.retrieval_requests

    assert len(retrieval_requests) == 1
    request = retrieval_requests[0]
    assert request.features_to_include == {"b"}

    assert set(filtered_job.request_result.feature_columns) == {"b"}
    assert set(filtered_job.request_result.entity_columns) == {"id"}
