from datetime import datetime

import pytest

from aladdin import Entity, FeatureStore, FeatureView, FeatureViewMetadata, FileSource, Float, Int32, String
from aladdin.feature_types import EventTimestamp


class BreastDiagnoseFeatureView(FeatureView):

    metadata = FeatureViewMetadata(
        name='breast_features',
        description='Features defining a scan and diagnose of potential cancer cells',
        tags={},
        batch_source=FileSource.csv_at(
            path='test_data/data-with-datetime.csv', mapping_keys={'id': 'scan_id'}
        ),
    )

    scan_id = Entity(dtype=Int32())

    created_at = EventTimestamp()

    diagnosis = String().description('The given diagnose. M for malignant, and B for benigne')
    is_malignant = (diagnosis == 'M').description('If the scanned cells was diagnosed as dangerous')

    radius_mean = Float()
    radius_se = Float()
    radius_worst = Float()

    texture_mean = Float()
    texture_se = Float()
    texture_worst = Float()

    perimeter_mean = Float()
    perimeter_se = Float()
    perimeter_worst = Float()

    area_mean = Float()
    area_se = Float()
    area_worst = Float()

    smoothness_mean = Float()
    smoothness_se = Float()
    smoothness_worst = Float()

    compactness_mean = Float()
    compactness_se = Float()
    compactness_worst = Float()

    concavity_mean = Float()
    concavity_se = Float()
    concavity_worst = Float()

    concave_points_mean = Float()
    concave_points_se = Float()
    concave_points_worst = Float()

    symmetry_mean = Float()
    symmetry_se = Float()
    symmetry_worst = Float()

    fractal_dimension_mean = Float()
    fractal_dimension_se = Float()
    fractal_dimension_worst = Float()


feature_view = BreastDiagnoseFeatureView()
store = FeatureStore.experimental()
store.add_feature_view(feature_view)


@pytest.mark.asyncio
async def test_between_datetime_features() -> None:
    features = await store.feature_view(feature_view.metadata.name).all().to_df()

    for feature in BreastDiagnoseFeatureView.select_all().features_to_include:
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
