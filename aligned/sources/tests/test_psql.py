from os import environ

import pytest

from aligned import FeatureStore, PostgreSQLConfig
from aligned.feature_view.feature_view import FeatureView
from conftest import DataTest
import platform


@pytest.fixture
def psql() -> PostgreSQLConfig:
    if 'PSQL_DATABASE_TEST' not in environ:
        environ['PSQL_DATABASE_TEST'] = 'postgresql://postgres:postgres@127.0.0.1:5433/aligned-test'

    return PostgreSQLConfig('PSQL_DATABASE_TEST')


@pytest.mark.skipif(
    platform.uname().machine.startswith('arm'), reason='Needs psycopg2 which is not supported on arm'
)
@pytest.mark.asyncio
async def test_postgresql(point_in_time_data_test: DataTest, psql: PostgreSQLConfig) -> None:

    psql_database = environ['PSQL_DATABASE_TEST']

    store = FeatureStore.experimental()

    for source in point_in_time_data_test.sources:
        view = source.view
        db_name = view.metadata.name
        source.data.to_pandas().to_sql(db_name, psql_database, if_exists='replace')

        view.metadata = FeatureView.metadata_with(  # type: ignore
            name=view.metadata.name,
            description=view.metadata.description,
            batch_source=psql.table(db_name),
        )
        store.add_feature_view(view)

    job = store.features_for(
        point_in_time_data_test.entities,
        point_in_time_data_test.feature_reference,
        event_timestamp_column='event_timestamp',
    )
    data = (await job.to_lazy_polars()).collect()

    expected = point_in_time_data_test.expected_output

    assert expected.shape == data.shape, f'Expected: {expected.shape}\nGot: {data.shape}'
    assert set(expected.columns) == set(data.columns), f'Expected: {expected.columns}\nGot: {data.columns}'

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.equals(expected), f'Expected: {expected}\nGot: {ordered_columns}'


@pytest.mark.skipif(
    platform.uname().machine.startswith('arm'), reason='Needs psycopg2 which is not supported on arm'
)
@pytest.mark.asyncio
async def test_postgresql_write(titanic_feature_store: FeatureStore, psql: PostgreSQLConfig) -> None:
    import polars as pl
    from polars.testing import assert_frame_equal

    source = psql.table('titanic')

    data: dict[str, list] = {'passenger_id': [1, 2, 3, 4], 'will_survive': [False, True, True, False]}

    store = titanic_feature_store.model('titanic').using_source(source)
    await store.insert_predictions(data)

    stored_data = await psql.fetch('SELECT * FROM titanic').to_lazy_polars()
    assert_frame_equal(
        pl.DataFrame(data),
        stored_data.collect(),
        check_row_order=False,
        check_column_order=False,
        check_dtype=False,
    )

    preds = await store.predictions_for({'passenger_id': [1, 3, 2, 4]}).to_lazy_polars()
    assert_frame_equal(
        pl.DataFrame(data),
        preds.collect(),
        check_row_order=False,
        check_column_order=False,
        check_dtype=False,
    )


@pytest.mark.skipif(
    platform.uname().machine.startswith('arm'), reason='Needs psycopg2 which is not supported on arm'
)
@pytest.mark.asyncio
async def test_postgresql_without_event(
    point_in_time_data_test_wituout_event_timestamp: DataTest, psql: PostgreSQLConfig
) -> None:

    psql_database = environ['PSQL_DATABASE_TEST']

    store = FeatureStore.experimental()

    for source in point_in_time_data_test_wituout_event_timestamp.sources:
        view = source.view
        db_name = view.metadata.name
        source.data.to_pandas().to_sql(db_name, psql_database, if_exists='replace')

        view.metadata = FeatureView.metadata_with(  # type: ignore
            name=view.metadata.name,
            description=view.metadata.description,
            batch_source=psql.table(db_name),
        )
        store.add_feature_view(view)

    job = store.features_for(
        point_in_time_data_test_wituout_event_timestamp.entities,
        point_in_time_data_test_wituout_event_timestamp.feature_reference,
    )
    data = (await job.to_lazy_polars()).collect()

    expected = point_in_time_data_test_wituout_event_timestamp.expected_output

    assert expected.shape == data.shape, f'Expected: {expected.shape}\nGot: {data.shape}'
    assert set(expected.columns) == set(data.columns), f'Expected: {expected.columns}\nGot: {data.columns}'

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.equals(expected), f'Expected: {expected}\nGot: {ordered_columns}'
