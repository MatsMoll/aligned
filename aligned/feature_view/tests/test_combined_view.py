import pytest

from aligned import FeatureStore


@pytest.mark.asyncio
async def test_combined_view(combined_feature_store: FeatureStore) -> None:

    entities = {'passenger_id': [1, 2, 3, 4, None], 'scan_id': [842302, 84300903, 843786, None, 842301]}
    result_job = combined_feature_store.features_for(
        entities,
        features=[
            'combined:some_feature',
            'combined:other_feature',
        ],
    )
    result = await result_job.log_each_job().to_pandas()

    assert 'some_feature' in result.columns
    assert 'other_feature' in result.columns

    assert result.shape == (len(entities['passenger_id']), 4)
    assert result.isna().sum().sum() == 4 + 2


@pytest.mark.asyncio
async def test_combined_view_get_all_features(combined_feature_store: FeatureStore) -> None:

    entities = {'passenger_id': [1, 2, 3, 4, None], 'scan_id': [842302, 84300903, 843786, None, 842301]}
    result = await combined_feature_store.features_for(entities, features=['combined:*']).to_pandas()

    assert 'some_feature' in result.columns
    assert 'other_feature' in result.columns

    assert result.shape == (len(entities['passenger_id']), 4)
    assert result.isna().sum().sum() == 4 + 2
