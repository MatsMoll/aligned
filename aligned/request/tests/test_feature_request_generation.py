import pytest

from aligned.feature_view.feature_view import FeatureView


@pytest.mark.asyncio
async def test_fetch_all_request(titanic_feature_view: FeatureView) -> None:

    compiled_view = type(titanic_feature_view).compile()
    request = compiled_view.request_all

    expected_features = {
        'age',
        'name',
        'sex',
        'survived',
        'sibsp',
        'cabin',
        'has_siblings',
        'is_male',
        'is_female',
        'is_mr',
    }

    assert not request.needs_event_timestamp
    assert len(request.needed_requests) == 1

    retrival_request = request.needed_requests[0]
    missing_features = expected_features - retrival_request.all_feature_names
    assert retrival_request.all_feature_names == expected_features, f'Missing features {missing_features}'
    assert retrival_request.entity_names == {'passenger_id'}

    assert len(request.request_result.entities) == 1
    assert len(request.request_result.features) == len(expected_features)


@pytest.mark.asyncio
async def test_fetch_features_request(titanic_feature_view: FeatureView) -> None:

    compiled_view = type(titanic_feature_view).compile()
    wanted_features = {'cabin', 'is_male'}
    request = compiled_view.request_for(wanted_features)
    expected_features = {'sex', 'cabin', 'is_male'}
    assert not request.needs_event_timestamp
    assert len(request.needed_requests) == 1

    retrival_request = request.needed_requests[0]
    missing_features = expected_features - retrival_request.all_feature_names
    # All the features to retrive and computed
    assert retrival_request.all_feature_names == expected_features, f'Missing features {missing_features}'
    assert retrival_request.entity_names == {'passenger_id'}

    # All the features that is returned
    assert len(request.request_result.entities) == 1
    assert len(request.request_result.features) == len(wanted_features)
