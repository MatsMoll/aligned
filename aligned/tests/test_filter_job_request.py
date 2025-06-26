import pytest
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


@pytest.mark.asyncio
async def test_filter_with_factory() -> None:
    from aligned import data_contract, String, Int16, InMemorySource

    @data_contract(
        source=InMemorySource.from_values(
            {
                "some_id": [1, 2, 3, 4, 5, 6],
                "x_value": [10, 10, 20, 20, 30, 30],
                "partition_value": ["a", "b", "a", "b", "a", "b"],
            }
        )
    )
    class TestData:
        some_id = Int16().as_entity()

        x_value = Int16()
        partition_value = String()

        other = x_value + 20

    schema = TestData()
    df = await TestData.query().filter(schema.partition_value == "a").to_polars()

    assert df.height == 3

    df = (
        await TestData.query()
        .filter((schema.x_value > 25) & (schema.partition_value == "b"))
        .to_polars()
    )

    schema.x_value.is_not_null

    assert df.height == 1
