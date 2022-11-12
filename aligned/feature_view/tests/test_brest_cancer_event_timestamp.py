from datetime import datetime

import pytest

from aligned import FeatureStore, FeatureView


@pytest.mark.asyncio
async def test_between_datetime_features(
    breast_scan_feature_view_with_datetime: FeatureView,
    breast_scan_with_timestamp_feature_store: FeatureStore,
) -> None:
    feature_view = breast_scan_feature_view_with_datetime
    store = breast_scan_with_timestamp_feature_store
    features = await store.feature_view(feature_view.metadata.name).all().to_df()

    for feature in type(feature_view).select_all().features_to_include:
        assert feature in features.columns

    assert 'created_at' in features.columns
    assert 'is_malignant' in features.columns
    assert 'diagnosis' in features.columns
    assert 'scan_id' in features.columns

    limit_features = (
        await store.feature_view(feature_view.metadata.name)
        .between(
            start_date=datetime(2020, 1, 5),
            end_date=datetime(2020, 1, 11),
        )
        .to_df()
    )

    assert limit_features.shape[0] == 6
    assert features['mean_fd_worst_for_group'].isna().count() != 0
