from datetime import timezone
import pytest
import polars as pl
from pathlib import Path

from aligned import ContractStore, FileSource, feature_view, Int32
from aligned.feature_view.feature_view import FeatureView
from aligned.retrival_job import RetrivalJob
from aligned.schemas.date_formatter import DateFormatter
from aligned.schemas.feature import Feature, FeatureLocation, FeatureType
from conftest import DataTest


@pytest.mark.asyncio
async def test_read_parquet(point_in_time_data_test: DataTest) -> None:

    store = ContractStore.experimental()

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
async def test_partition_parquet(point_in_time_data_test: DataTest) -> None:
    from datetime import datetime

    store = ContractStore.experimental()

    agg_features: list[str] = []

    for source in point_in_time_data_test.sources:
        view = source.view
        view_name = view.metadata.name

        compiled = view.compile()

        if '_agg' in view_name:
            agg_features.extend([feat.name for feat in compiled.aggregated_features])
            continue

        entities = compiled.entitiy_names
        partition_keys = list(entities)

        file_source = FileSource.partitioned_parquet_at(
            f'test_data/temp/{view_name}', partition_keys=partition_keys
        )
        await file_source.delete()
        await file_source.write_polars(source.data.lazy())

        view.metadata = FeatureView.metadata_with(  # type: ignore
            name=view.metadata.name,
            description=view.metadata.description,
            batch_source=file_source,
        )
        store.add_feature_view(view)

    job = store.features_for(
        point_in_time_data_test.entities,
        [feat for feat in point_in_time_data_test.feature_reference if '_agg' not in feat],
        event_timestamp_column='event_timestamp',
    )
    data = (await job.to_lazy_polars()).collect()

    expected = point_in_time_data_test.expected_output.drop(agg_features)
    assert expected.shape == data.shape, f'Expected: {expected.shape}\nGot: {data.shape}'
    assert set(expected.columns) == set(data.columns), f'Expected: {expected.columns}\nGot: {data.columns}'

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.equals(expected), f'Expected: {expected}\nGot: {ordered_columns}'

    await store.upsert_into(
        FeatureLocation.feature_view('loan'),
        pl.DataFrame(
            {
                'loan_amount': [4000],
                'loan_id': [10000],
                'event_timestamp': [datetime.now(tz=timezone.utc)],
                'personal_income': [59000],
                'loan_status': [1],
            }
        ),
    )
    await store.upsert_into(
        FeatureLocation.feature_view('loan'),
        pl.DataFrame(
            {
                'loan_id': [10001],
                'loan_amount': [4000],
                'event_timestamp': [datetime.now(tz=timezone.utc)],
                'personal_income': [59000],
                'loan_status': [1],
            }
        ),
    )
    await store.feature_view('loan').all().to_polars()


@pytest.mark.asyncio
async def test_partition_parquet_upsert():
    from polars.testing import assert_frame_equal
    from aligned import FileSource
    from aligned.request.retrival_request import RetrivalRequest

    dir = FileSource.directory('test_data/temp')
    source = dir.partitioned_parquet_at('partition_upsert', partition_keys=['a', 'b'])

    request = RetrivalRequest(
        name='test',
        location=FeatureLocation.feature_view('test'),
        entities={Feature('e', FeatureType.int8())},
        features={
            Feature('a', FeatureType.int8()),
            Feature('b', FeatureType.int8()),
            Feature('x', FeatureType.int8()),
        },
        derived_features=set(),
    )
    initial_data = pl.DataFrame(
        {'a': [1, 1, 1, 2, 2, 2], 'b': [1, 2, 3, 1, 2, 3], 'e': [1, 2, 3, 4, 5, 6], 'x': [1, 2, 3, 4, 5, 6]}
    )
    new_data = pl.DataFrame({'a': [1, 1, 1, 1], 'b': [2, 2, 3, 1], 'e': [1, 2, 3, 7], 'x': [7, 8, 9, 10]})
    expected = pl.concat([new_data, initial_data]).unique(['e'], keep='first')

    await source.delete()
    await source.overwrite(RetrivalJob.from_polars_df(initial_data, [request]), request)
    data = await source.to_polars()
    assert_frame_equal(data, initial_data, check_column_order=False)

    await source.upsert(RetrivalJob.from_polars_df(new_data, [request]), request)
    new = await source.to_polars()
    assert_frame_equal(new.sort('e'), expected.sort('e'), check_column_order=False)


@pytest.mark.asyncio
async def test_parquet(point_in_time_data_test: DataTest) -> None:

    store = ContractStore.experimental()

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

    store = ContractStore.experimental()

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

    store = ContractStore.experimental()

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
        compiled = view.compile()
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
