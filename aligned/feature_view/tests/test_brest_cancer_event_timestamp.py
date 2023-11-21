from datetime import datetime

import pytest

from aligned import FeatureStore
from aligned.feature_view.feature_view import FeatureView


@pytest.mark.asyncio
async def test_between_datetime_features(
    breast_scan_feature_view_with_datetime: FeatureView,
    breast_scan_with_timestamp_feature_store: FeatureStore,
) -> None:
    feature_view = breast_scan_feature_view_with_datetime
    store = breast_scan_with_timestamp_feature_store
    features = await store.feature_view(feature_view.metadata.name).all().to_pandas()

    assert 'created_at' in features.columns
    assert 'is_malignant' in features.columns
    assert 'diagnosis' in features.columns
    assert 'scan_id' in features.columns

    limit_features = (
        await store.feature_view(feature_view.metadata.name)
        .between_dates(
            start_date=datetime(2020, 1, 5),
            end_date=datetime(2020, 1, 11),
        )
        .to_pandas()
    )
    assert limit_features.shape[0] == 6


@pytest.mark.asyncio
async def test_between_datetime_features_with_aggregation(
    breast_scan_feature_view_with_datetime_and_aggregation: FeatureView,
    breast_scan_with_timestamp_and_aggregation_feature_store: FeatureStore,
) -> None:
    feature_view = breast_scan_feature_view_with_datetime_and_aggregation
    store = breast_scan_with_timestamp_and_aggregation_feature_store
    features = await store.feature_view(feature_view.metadata.name).all().to_pandas()

    assert 'created_at' in features.columns
    assert 'is_malignant' in features.columns
    assert 'diagnosis' in features.columns
    assert 'scan_id' in features.columns

    limit_features = (
        await store.feature_view(feature_view.metadata.name)
        .between_dates(
            start_date=datetime(2020, 1, 5),
            end_date=datetime(2020, 1, 11),
        )
        .to_pandas()
    )

    assert limit_features.shape[0] == 6


# @pytest.mark.asyncio
# async def test_between_datetime_features_polars(
#     breast_scan_feature_view_with_datetime: FeatureView,
#     breast_scan_with_timestamp_feature_store: FeatureStore,
# ) -> None:
#     feature_view = breast_scan_feature_view_with_datetime
#     store = breast_scan_with_timestamp_feature_store
#     job = store.feature_view(feature_view.metadata.name).all()
#     features = (await job.to_polars()).collect()

#     assert 'created_at' in features.columns
#     assert 'is_malignant' in features.columns
#     assert 'diagnosis' in features.columns
#     assert 'scan_id' in features.columns

#     limit_features = (
#         await store.feature_view(feature_view.metadata.name)
#         .between_dates(
#             start_date=datetime(2020, 1, 5),
#             end_date=datetime(2020, 1, 11),
#         )
#         .to_polars()
#     ).collect()

#     assert limit_features.shape[0] == 6
