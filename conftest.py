import pytest

from aladdin import Bool, Entity, FeatureView, FeatureViewMetadata, FileSource, Float, Int32, String
from aladdin.feature_store import FeatureStore
from aladdin.local.source import CsvFileSource


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
        scaled_age = age.standard_scaled()
        is_mr = name.contains('Mr.')

    return TitanicPassenger()


@pytest.fixture
def titanic_feature_store(titanic_feature_view: FeatureView) -> FeatureStore:
    feature_store = FeatureStore.experimental()
    feature_store.add_feature_view(titanic_feature_view)
    return feature_store
