import pytest
import polars as pl
from pathlib import Path

from aligned import FeatureStore, FileSource, feature_view, Int32
from aligned.feature_view.feature_view import FeatureView
from aligned.schemas.date_formatter import DateFormatter
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

        stored = await store.feature_view(view.metadata.name).all().to_lazy_polars()
        df = stored.select(source.data.columns).collect()
        assert df.equals(source.data)


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
    data = (await job.to_lazy_polars()).collect()

    expected = point_in_time_data_test.expected_output

    assert expected.shape == data.shape, f'Expected: {expected.shape}\nGot: {data.shape}'
    assert set(expected.columns) == set(data.columns), f'Expected: {expected.columns}\nGot: {data.columns}'

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.equals(expected), f'Expected: {expected}\nGot: {ordered_columns}'


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
    data = (await job.to_lazy_polars()).collect()

    expected = point_in_time_data_test_wituout_event_timestamp.expected_output

    assert expected.shape == data.shape, f'Expected: {expected.shape}\nGot: {data.shape}'
    assert set(expected.columns) == set(data.columns), f'Expected: {expected.columns}\nGot: {data.columns}'

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.equals(expected), f'Expected: {expected}\nGot: {ordered_columns}'


@pytest.mark.asyncio
async def test_read_csv(point_in_time_data_test: DataTest) -> None:

    store = FeatureStore.experimental()

    for source in point_in_time_data_test.sources:
        view = source.view
        view_name = view.metadata.name
        if '_agg' in view_name:
            continue

        file_source = FileSource.csv_at(
            f'test_data/{view_name}.csv', date_formatter=DateFormatter.unix_timestamp()
        )

        view.metadata = FeatureView.metadata_with(  # type: ignore
            name=view.metadata.name,
            description=view.metadata.description,
            batch_source=file_source,
        )
        compiled = view.compile_instance()
        assert compiled.source.path == file_source.path

        store.add_compiled_view(compiled)

        Path(file_source.path).unlink(missing_ok=True)

        await store.feature_view(compiled.name).insert(
            store.feature_view(compiled.name).process_input(source.data)
        )

        csv = pl.read_csv(file_source.path)
        schemas = dict(csv.schema)

        for feature in view.compile().request_all.request_result.features:
            if feature.dtype.name == 'datetime':
                assert schemas[feature.name].is_numeric()

        # Polars
        stored = await store.feature_view(compiled.name).all().to_polars()
        df = stored.select(source.data.columns)
        assert df.equals(source.data)


@pytest.mark.asyncio
async def test_read_optional_csv() -> None:

    source = FileSource.csv_at('test_data/optional_test.csv')
    df = pl.DataFrame(
        {
            'a': [1, 2, 3],
            'c': [1, 2, 3],
        }
    )
    await source.write_polars(df.lazy())

    @feature_view(name='test', source=source)
    class Test:
        a = Int32().as_entity()
        b = Int32().is_optional()
        c = Int32()

        filled = b.fill_na(0)

    expected_df = df.with_columns(pl.lit(None).alias('b'), pl.lit(0).alias('filled'))
    loaded = await Test.query().all().to_polars()  # type: ignore

    assert loaded.equals(expected_df.select(loaded.columns))

    facts = await Test.query().features_for({'a': [2]}).to_polars()  # type: ignore
    assert expected_df.filter(pl.col('a') == 2).equals(facts.select(expected_df.columns))


@pytest.mark.asyncio
async def test_read_optional_view() -> None:

    source = FileSource.csv_at('test_data/optional_test.csv')
    df = pl.DataFrame(
        {
            'a': [1, 2, 3],
            'c': [1, 2, 3],
        }
    )
    await source.write_polars(df.lazy())

    @feature_view(name='test_a', source=source)
    class TestA:
        a = Int32().as_entity()
        c = Int32()

    @feature_view(name='test', source=TestA)  # type: ignore
    class Test:
        a = Int32().as_entity()
        b = Int32().is_optional()
        c = Int32()

        filled = b.fill_na(0)

    expected_df = df.with_columns(pl.lit(None).alias('b'), pl.lit(0).alias('filled'))
    loaded = await Test.query().all().to_polars()  # type: ignore

    assert loaded.equals(expected_df.select(loaded.columns))

    facts = await Test.query().features_for({'a': [2]}).to_polars()  # type: ignore
    assert expected_df.filter(pl.col('a') == 2).equals(facts.select(expected_df.columns))
