import pytest

from aligned import Entity, FeatureStore, FeatureView, FeatureViewMetadata, FileSource, Float, Int32, String
from aligned.online_source import InMemoryOnlineSource
from aligned.schemas.repo_definition import RepoDefinition


class BreastDiagnoseFeatureView(FeatureView):

    metadata = FeatureViewMetadata(
        name='breast_features',
        description='Features defining a scan and diagnose of potential cancer cells',
        tags={},
        batch_source=FileSource.csv_at(path='test_data/data.csv', mapping_keys={'id': 'scan_id'}),
    )

    scan_id = Entity(dtype=Int32())

    diagnosis = String().description('The given diagnose. M for malignant, and B for benigne')
    is_malignant = (diagnosis == 'M').description('If the scanned cells was diagnosed as dangerous')

    radius_mean = Float()
    radius_se = Float()
    radius_worst = Float()
    scaled_mean_radius = radius_mean.standard_scaled()

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


@pytest.mark.asyncio
async def test_standard_scaled_feature() -> None:
    feature_view = BreastDiagnoseFeatureView()
    store = FeatureStore.experimental()
    await store.add_feature_view(feature_view)

    features = await store.feature_view(feature_view.metadata.name).all().to_df()

    for feature in BreastDiagnoseFeatureView.select_all().features_to_include:
        assert feature in features.columns

    assert 'scaled_mean_radius' in features.columns
    assert not features['scaled_mean_radius'].isna().any()


@pytest.mark.asyncio
async def test_online_store_standard_scaling() -> None:
    compiled_view = await BreastDiagnoseFeatureView.compile()

    definition = RepoDefinition(
        feature_views={compiled_view},
        combined_feature_views=set(),
        models={},
        online_source=InMemoryOnlineSource(),
    )
    store = FeatureStore.from_definition(definition)

    await store.feature_view(compiled_view.name).write(
        {
            'radius_mean': [17.99, 14.127291739894552],
            'scan_id': [10, 11],
        }
    )

    stored = await store.features_for(
        {'scan_id': [10, 11]}, features=[f'{compiled_view.name}:scaled_mean_radius']
    ).to_df()

    assert ((stored['scaled_mean_radius'].values * 1000).astype(int) == [1096, 0]).all()
