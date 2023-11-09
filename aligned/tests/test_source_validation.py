import pytest

from aligned import FeatureStore, FileSource
from aligned.schemas.feature import FeatureType, FeatureLocation
from aligned.source_validation import validate_sources_in


@pytest.mark.asyncio
async def test_source_validation(titanic_feature_store: FeatureStore) -> None:

    source = FileSource.parquet_at('test_data/titanic.parquet')

    views = titanic_feature_store.views_with_config(source)

    assert len(views) == 1
    validation = await validate_sources_in(views)

    assert {FeatureLocation.feature_view('titanic_parquet'): True} == validation


# @pytest.mark.asyncio
# async def test_source_validation_psql(titanic_feature_view: FeatureView) -> None:
#
#     if 'PSQL_DATABASE_TEST' not in environ:
#         environ['PSQL_DATABASE_TEST'] = 'postgresql://postgres:postgres@localhost:5433/aligned-test'
#
#     psql_config = PostgreSQLConfig('PSQL_DATABASE_TEST')
#     titanic_feature_view.metadata.source = psql_config.table('titanic')
#
#     store = FeatureStore.experimental()
#     store.add_feature_view(titanic_feature_view)
#     views = store.views_with_config(psql_config)
#
#     assert len(views) == 1
#     validation = await validate_sources_in(views)
#
#     assert {FeatureLocation.feature_view('titanic'): False} == validation


@pytest.mark.asyncio
async def test_schema_loading() -> None:
    source = FileSource.parquet_at('test_data/titanic.parquet')
    schema = await source.schema()
    dtype_schema = {key: feature.dtype for key, feature in schema.items()}
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
