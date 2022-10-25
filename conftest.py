from math import ceil, floor

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
from aladdin.compiler.transformation_factory import FillNaStrategy
from aladdin.feature_store import FeatureStore
from aladdin.feature_view.combined_view import CombinedFeatureView, CombinedFeatureViewMetadata
from aladdin.local.source import CsvFileSource


@pytest.fixture
def scan_without_datetime() -> CsvFileSource:
    return FileSource.csv_at(path='test_data/data.csv', mapping_keys={'id': 'scan_id'})


@pytest.fixture
def breast_scan_feature_viewout_with_datetime(scan_without_datetime: CsvFileSource) -> FeatureView:
    class BreastDiagnoseFeatureView(FeatureView):

        metadata = FeatureViewMetadata(
            name='breast_features',
            description='Features defining a scan and diagnose of potential cancer cells',
            tags={},
            batch_source=scan_without_datetime,
        )

        scan_id = Entity(dtype=Int32())
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
async def breast_scan_without_timestamp_feature_store(breast_scan_feature_viewout_with_datetime: FeatureView):
    store = FeatureStore.experimental()
    await store.add_feature_view(breast_scan_feature_viewout_with_datetime)
    return store


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
async def breast_scan_with_timestamp_feature_store(breast_scan_feature_view_with_datetime: FeatureView):
    store = FeatureStore.experimental()
    await store.add_feature_view(breast_scan_feature_view_with_datetime)
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
        age = (
            Float().is_required().lower_bound(0).upper_bound(100).description('A float as some have decimals')
        )

        name = String()
        sex = String().accepted_values(['male', 'female'])
        survived = Bool().description('If the passenger survived')

        sibsp = (
            Int32()
            .lower_bound(0, is_inclusive=True)
            .upper_bound(20, is_inclusive=True)
            .description('Number of siblings on titanic')
        )

        cabin = String()

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        scaled_age = age.standard_scaled(limit=100)
        is_mr = name.contains('Mr.')

    return TitanicPassenger()


@pytest.fixture
async def titanic_feature_store(titanic_feature_view: FeatureView) -> FeatureStore:
    feature_store = FeatureStore.experimental()
    await feature_store.add_feature_view(titanic_feature_view)
    return feature_store


@pytest.fixture
def alot_of_transforations_feature_view(titanic_source: CsvFileSource) -> FeatureView:
    class TitanicPassenger(FeatureView):

        metadata = FeatureViewMetadata(
            name='titanic', description='Some features from the titanic dataset', batch_source=titanic_source
        )

        passenger_id = Entity(dtype=Int32())

        # Input values
        age = Float()
        name = String()
        sex = String()
        survived = Bool()
        sibsp = Int32()
        cabin = String().fill_na('Nada')

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        ordinal_sex = sex.ordinal_categories(['male', 'female'])
        scaled_age = age.standard_scaled(limit=100)
        filled_age = age.fill_na(FillNaStrategy.mean(limit=100))
        is_mr = name.contains('Mr.')

        ratio = scaled_age / age
        floor_ratio = scaled_age // age
        adding = sibsp + age
        subtracting = sibsp - age
        floored_age = floor(age)
        ceiled_age = ceil(age)
        rounded_age = round(age)
        abs_scaled_age = abs(scaled_age)

        inverted_is_mr = ~is_mr
        logical_and = is_mr & survived
        logical_or = is_mr | survived

    return TitanicPassenger()


@pytest.fixture
async def alot_of_transforation_feature_store(
    alot_of_transforations_feature_view: FeatureView,
) -> FeatureStore:
    feature_store = FeatureStore.experimental()
    await feature_store.add_feature_view(alot_of_transforations_feature_view)
    return feature_store


@pytest.fixture
async def combined_view(
    titanic_feature_view, breast_scan_feature_viewout_with_datetime
) -> CombinedFeatureView:
    class SomeCombinedView(CombinedFeatureView):

        metadata = CombinedFeatureViewMetadata(
            name='combined', description='Some features that depend on multiple view'
        )

        titanic = titanic_feature_view
        cancer_scan = breast_scan_feature_viewout_with_datetime

        some_feature = titanic.age + cancer_scan.radius_mean
        other_feature = titanic.scaled_age + cancer_scan.radius_mean

    return SomeCombinedView()


@pytest.fixture
async def combined_feature_store(
    titanic_feature_view: FeatureView,
    breast_scan_feature_viewout_with_datetime: FeatureView,
    combined_view: CombinedFeatureView,
) -> FeatureStore:
    feature_store = FeatureStore.experimental()
    await feature_store.add_feature_view(titanic_feature_view)
    await feature_store.add_feature_view(breast_scan_feature_viewout_with_datetime)
    await feature_store.add_combined_feature_view(combined_view)
    return feature_store
