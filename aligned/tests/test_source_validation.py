import pytest

from aligned import ContractStore, FileSource
from aligned.schemas.feature import FeatureType, FeatureLocation
from aligned.source_validation import validate_sources_in


@pytest.mark.asyncio
async def test_source_validation(titanic_feature_store: ContractStore) -> None:

    source = FileSource.parquet_at('test_data/titanic.parquet')

    views = titanic_feature_store.views_with_config(source)

    assert len(views) == 1
    validation = await validate_sources_in(views)

    assert {FeatureLocation.feature_view('titanic_parquet'): True} == validation


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
        'age': FeatureType(name='float'),
        'sibsp': FeatureType(name='int64'),
        'Parch': FeatureType(name='int64'),
        'Ticket': FeatureType(name='string'),
        'Fare': FeatureType(name='float'),
        'cabin': FeatureType(name='string'),
        'Embarked': FeatureType(name='string'),
    }


@pytest.mark.asyncio
async def test_feature_view_generation() -> None:
    fv_impl = await FileSource.csv_at('test_data/data.csv').feature_view_code('my_view')
    assert '' in fv_impl
