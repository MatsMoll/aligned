import pytest

from aligned import FileSource
from aligned.schemas.feature import FeatureType


@pytest.mark.asyncio
async def test_schema_loading() -> None:
    source = FileSource.parquet_at('test_data/titanic.parquet')
    dtype_schema = await source.schema()
    assert dtype_schema == {
        'passenger_id': FeatureType(name='int64'),
        'survived': FeatureType(name='int64'),
        'Pclass': FeatureType(name='int64'),
        'name': FeatureType(name='string'),
        'sex': FeatureType(name='string'),
        'age': FeatureType(name='float64'),
        'sibsp': FeatureType(name='int64'),
        'Parch': FeatureType(name='int64'),
        'Ticket': FeatureType(name='string'),
        'Fare': FeatureType(name='float64'),
        'cabin': FeatureType(name='string'),
        'Embarked': FeatureType(name='string'),
    }
