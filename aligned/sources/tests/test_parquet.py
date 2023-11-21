import pytest

from aligned import FeatureStore, FileSource
from aligned.feature_view.feature_view import FeatureView
from conftest import DataTest


@pytest.mark.asyncio
async def test_read_parquet(point_in_time_data_test: DataTest) -> None:

    store = FeatureStore.experimental()

    for source in point_in_time_data_test.sources:
        view = source.view
        view_name = view.metadata.name
        if '_agg' in view_name:
            continue

        file_source = FileSource.parquet_at(f'test_data/{view_name}.parquet')
        await file_source.write_polars(source.data.lazy())

        view.metadata = FeatureView.metadata_with(  # type: ignore
            name=view.metadata.name,
            description=view.metadata.description,
            batch_source=file_source,
        )
        store.add_feature_view(view)

        stored = await store.feature_view(view.metadata.name).all().to_polars()
        df = stored.select(source.data.columns).collect()
        assert df.frame_equal(source.data)


@pytest.mark.asyncio
async def test_parquest(point_in_time_data_test: DataTest) -> None:

    store = FeatureStore.experimental()

    for source in point_in_time_data_test.sources:
        view = source.view
        view_name = view.metadata.name

        file_source = FileSource.parquet_at(f'test_data/{view_name}.parquet')
        await file_source.write_polars(source.data.lazy())

        view.metadata = FeatureView.metadata_with(  # type: ignore
            name=view.metadata.name,
            description=view.metadata.description,
            batch_source=file_source,
        )
        store.add_feature_view(view)

    job = store.features_for(
        point_in_time_data_test.entities,
        point_in_time_data_test.feature_reference,
        event_timestamp_column='event_timestamp',
    )
    data = (await job.to_polars()).collect()

    expected = point_in_time_data_test.expected_output

    assert expected.shape == data.shape, f'Expected: {expected.shape}\nGot: {data.shape}'
    assert set(expected.columns) == set(data.columns), f'Expected: {expected.columns}\nGot: {data.columns}'

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.frame_equal(expected), f'Expected: {expected}\nGot: {ordered_columns}'


@pytest.mark.asyncio
async def test_parquet_without_event_timestamp(
    point_in_time_data_test_wituout_event_timestamp: DataTest,
) -> None:

    store = FeatureStore.experimental()

    for source in point_in_time_data_test_wituout_event_timestamp.sources:
        view = source.view
        view_name = view.metadata.name

        file_source = FileSource.parquet_at(f'test_data/{view_name}.parquet')
        await file_source.write_polars(source.data.lazy())

        view.metadata = FeatureView.metadata_with(  # type: ignore
            name=view.metadata.name,
            description=view.metadata.description,
            batch_source=file_source,
        )
        store.add_feature_view(view)

    job = store.features_for(
        point_in_time_data_test_wituout_event_timestamp.entities,
        point_in_time_data_test_wituout_event_timestamp.feature_reference,
    )
    data = (await job.to_polars()).collect()

    expected = point_in_time_data_test_wituout_event_timestamp.expected_output

    assert expected.shape == data.shape, f'Expected: {expected.shape}\nGot: {data.shape}'
    assert set(expected.columns) == set(data.columns), f'Expected: {expected.columns}\nGot: {data.columns}'

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.frame_equal(expected), f'Expected: {expected}\nGot: {ordered_columns}'
