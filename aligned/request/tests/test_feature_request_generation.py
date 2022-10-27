import pytest

from aligned import FeatureView


@pytest.mark.asyncio
async def test_fetch_all_request(titanic_feature_view: FeatureView) -> None:

    compiled_view = await type(titanic_feature_view).compile()
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
        'scaled_age',
        'is_mr',
    }

    assert not request.needs_event_timestamp
    assert len(request.needed_requests) == 1

    retrival_request = request.needed_requests[0]
    assert retrival_request.all_feature_names == expected_features
    assert retrival_request.entity_names == {'passenger_id'}

    assert len(request.request_result.entities) == 1
    assert len(request.request_result.features) == len(expected_features)


@pytest.mark.asyncio
async def test_fetch_features_request(titanic_feature_view: FeatureView) -> None:

    compiled_view = await type(titanic_feature_view).compile()
    wanted_features = {'cabin', 'scaled_age', 'is_male'}
    request = compiled_view.request_for(wanted_features)
    expected_features = {'age', 'sex', 'cabin', 'is_male', 'scaled_age'}
    assert not request.needs_event_timestamp
    assert len(request.needed_requests) == 1

    retrival_request = request.needed_requests[0]
    assert retrival_request.all_feature_names == expected_features
    assert retrival_request.entity_names == {'passenger_id'}

    assert len(request.request_result.entities) == 1
    assert len(request.request_result.features) == len(expected_features)
