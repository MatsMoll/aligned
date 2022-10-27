import pytest

from aligned import FeatureStore
from aligned.exceptions import CombinedFeatureViewQuerying


def test_combined_view_error(combined_feature_store: FeatureStore) -> None:

    with pytest.raises(CombinedFeatureViewQuerying):
        combined_feature_store.feature_view('combined')


@pytest.mark.asyncio
async def test_combined_view(combined_feature_store: FeatureStore) -> None:

    entities = {'passenger_id': [1, 2, 3, 4, None], 'scan_id': [842302, 84300903, 843786, None, 842301]}
    result = await combined_feature_store.features_for(
        entities,
        features=[
            'combined:some_feature',
            'combined:other_feature',
        ],
    ).to_df()

    assert result.shape == (len(entities['passenger_id']), 4)
    assert result.isna().sum().sum() == 4 + 2


@pytest.mark.asyncio
async def test_combined_view_get_all_features(combined_feature_store: FeatureStore) -> None:

    entities = {'passenger_id': [1, 2, 3, 4, None], 'scan_id': [842302, 84300903, 843786, None, 842301]}
    result = await combined_feature_store.features_for(entities, features=['combined:*']).to_df()

    assert result.shape == (len(entities['passenger_id']), 4)
    assert result.isna().sum().sum() == 4 + 2
