import pytest

from aligned import ContractStore
from aligned.feature_view.feature_view import FeatureView


@pytest.mark.asyncio
async def test_all_features(
    breast_scan_without_timestamp_feature_store: ContractStore,
    breast_scan_feature_viewout_with_datetime: FeatureView,
) -> None:
    store = breast_scan_without_timestamp_feature_store
    feature_view = breast_scan_feature_viewout_with_datetime

    features = await store.feature_view(feature_view.metadata.name).all().to_pandas()

    assert 'is_malignant' in features.columns
    assert not features['is_malignant'].isna().any()
    assert 'diagnosis' in features.columns
    assert 'scan_id' in features.columns

    limit = 10
    limit_features = await store.feature_view(feature_view.metadata.name).all(limit=limit).to_pandas()

    assert limit_features.shape[0] == limit
