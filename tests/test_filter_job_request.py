from aligned.retrival_job import RetrivalJob, RetrivalRequest
from aligned.schemas.feature import Feature, FeatureLocation, FeatureType


def test_filter_job_result_request() -> None:
    job = RetrivalJob.from_dict(
        {'id': [1], 'a': [3], 'b': ['test'], 'c': [10]},
        request=RetrivalRequest(
            name='test',
            location=FeatureLocation.feature_view('test'),
            entities={Feature('id', FeatureType.int32())},
            features={
                Feature('a', FeatureType.int32()),
                Feature('b', FeatureType.string()),
                Feature('c', FeatureType.int32()),
            },
            derived_features=set(),
        ),
    )

    assert set(job.request_result.feature_columns) == {'a', 'b', 'c'}
    assert set(job.request_result.entity_columns) == {'id'}
    filtered_job = job.select_columns({'b'})
    assert set(filtered_job.request_result.feature_columns) == {'b'}
    assert set(filtered_job.request_result.entity_columns) == {'id'}


def test_filter_job_retrival_requests() -> None:
    job = RetrivalJob.from_dict(
        {'id': [1], 'a': [3], 'b': ['test'], 'c': [10]},
        request=RetrivalRequest(
            name='test',
            location=FeatureLocation.feature_view('test'),
            entities={Feature('id', FeatureType.int32())},
            features={
                Feature('a', FeatureType.int32()),
                Feature('b', FeatureType.string()),
                Feature('c', FeatureType.int32()),
            },
            derived_features=set(),
        ),
    )

    retrival_requests = job.retrival_requests

    assert len(retrival_requests) == 1
    request = retrival_requests[0]

    assert request.features_to_include == {'b', 'a', 'c'}

    assert set(job.request_result.feature_columns) == {'a', 'b', 'c'}
    assert set(job.request_result.entity_columns) == {'id'}

    filtered_job = job.select_columns({'b'})
    retrival_requests = filtered_job.retrival_requests

    assert len(retrival_requests) == 1
    request = retrival_requests[0]
    assert request.features_to_include == {'b'}

    assert set(filtered_job.request_result.feature_columns) == {'b'}
    assert set(filtered_job.request_result.entity_columns) == {'id'}
