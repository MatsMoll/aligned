from os import environ

import pytest

from aligned import FeatureStore, FeatureView, PostgreSQLConfig
from conftest import DataTest
import platform


@pytest.mark.skipif(
    platform.uname().machine.startswith('arm'), reason='Needs psycopg2 which is not supported on arm'
)
@pytest.mark.asyncio
async def test_postgresql(point_in_time_data_test: DataTest) -> None:

    if 'PSQL_DATABASE_TEST' not in environ:
        environ['PSQL_DATABASE_TEST'] = 'postgresql://postgres:postgres@127.0.0.1:5433/aligned-test'

    psql_database = environ['PSQL_DATABASE_TEST']

    store = FeatureStore.experimental()

    for source in point_in_time_data_test.sources:
        view = source.view
        db_name = view.metadata.name
        source.data.to_pandas().to_sql(db_name, psql_database, if_exists='replace')

        view.metadata = FeatureView.metadata_with(  # type: ignore
            name=view.metadata.name,
            description=view.metadata.description,
            batch_source=PostgreSQLConfig('PSQL_DATABASE_TEST').table(db_name),
        )
        store.add_feature_view(view)

    job = store.features_for(
        point_in_time_data_test.entities.to_dict(as_series=False), point_in_time_data_test.feature_reference
    )
    data = (await job.to_polars()).collect()

    expected = point_in_time_data_test.expected_output

    assert expected.shape == data.shape, f'Expected: {expected.shape}\nGot: {data.shape}'
    assert set(expected.columns) == set(data.columns), f'Expected: {expected.columns}\nGot: {data.columns}'

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.frame_equal(expected), f'Expected: {expected}\nGot: {ordered_columns}'
