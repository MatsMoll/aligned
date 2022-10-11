import pytest

from aladdin import (
    Bool,
    Entity,
    EventTimestamp,
    FeatureView,
    FeatureViewMetadata,
    FileSource,
    Float,
    Int32,
    String,
)
from aladdin.feature_store import FeatureStore
from aladdin.local.source import CsvFileSource


@pytest.fixture
def scan_with_datetime() -> CsvFileSource:
    return FileSource.csv_at(path='test_data/data-with-datetime.csv', mapping_keys={'id': 'scan_id'})


@pytest.fixture
def breast_scan_feature_view_with_datetime(scan_with_datetime: CsvFileSource) -> FeatureView:
    class BreastDiagnoseFeatureView(FeatureView):

        metadata = FeatureViewMetadata(
            name='breast_features',
            description='Features defining a scan and diagnose of potential cancer cells',
            tags={},
            batch_source=scan_with_datetime,
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

    return BreastDiagnoseFeatureView()


@pytest.fixture
def breast_scan_feature_store(breast_scan_feature_view_with_datetime: FeatureView):
    store = FeatureStore.experimental()
    store.add_feature_view(breast_scan_feature_view_with_datetime)
    return store


@pytest.fixture
def titanic_source() -> CsvFileSource:
    return FileSource.csv_at(
        'test_data/titanic_dataset.csv',
        mapping_keys={
            'PassengerId': 'passenger_id',
            'Age': 'age',
            'Name': 'name',
            'Sex': 'sex',
            'Survived': 'survived',
            'SibSp': 'sibsp',
            'Cabin': 'cabin',
        },
    )


@pytest.fixture
def titanic_feature_view(titanic_source: CsvFileSource) -> FeatureView:
    class TitanicPassenger(FeatureView):

        metadata = FeatureViewMetadata(
            name='titanic', description='Some features from the titanic dataset', batch_source=titanic_source
        )

        passenger_id = Entity(dtype=Int32())

        # Input values
        age = Float().description('A float as some have decimals')
        name = String()
        sex = String()
        survived = Bool().description('If the passenger survived')
        sibsp = Int32().description('Number of siblings on titanic')
        cabin = String()

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        scaled_age = age.standard_scaled(limit=100)
        is_mr = name.contains('Mr.')

    return TitanicPassenger()


@pytest.fixture
def titanic_feature_store(titanic_feature_view: FeatureView) -> FeatureStore:
    feature_store = FeatureStore.experimental()
    feature_store.add_feature_view(titanic_feature_view)
    return feature_store
