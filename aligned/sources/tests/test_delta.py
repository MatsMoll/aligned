from datetime import timezone
import pytest
import polars as pl
from pathlib import Path

from aligned import ContractStore, FileSource, feature_view, Int32
from aligned.feature_view.feature_view import FeatureView
from aligned.retrieval_job import RetrievalJob
from aligned.schemas.date_formatter import DateFormatter
from aligned.schemas.feature import Feature, FeatureLocation, FeatureType
from aligned.schemas.transformation import Expression
from aligned.sources.local import DeltaConfig, DeltaFileSource
from conftest import DataTest


@pytest.mark.asyncio
async def test_read_delta(point_in_time_data_test: DataTest) -> None:
    store = ContractStore.experimental()

    for source in point_in_time_data_test.sources:
        view = source.view
        view_name = view.metadata.name
        if "_agg" in view_name:
            continue

        file_source = FileSource.delta_at(f"test_data/temp/delta/{view_name}")

        await file_source.delete()
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
async def test_upsert_delta(point_in_time_data_test: DataTest) -> None:
    from datetime import datetime

    store = ContractStore.experimental()

    agg_features: list[str] = []

    for source in point_in_time_data_test.sources:
        view = source.view
        view_name = view.metadata.name

        compiled = view.compile()

        if "_agg" in view_name:
            agg_features.extend([feat.name for feat in compiled.aggregated_features])
            continue

        file_source = FileSource.delta_at(f"test_data/temp/delta/{view_name}")

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
        [
            feat
            for feat in point_in_time_data_test.feature_reference
            if "_agg" not in feat
        ],
        event_timestamp_column="event_timestamp",
    )

    data = (await job.to_lazy_polars()).collect()

    expected = point_in_time_data_test.expected_output.drop(agg_features)
    assert (
        expected.shape == data.shape
    ), f"Expected: {expected.shape}\nGot: {data.shape}"
    assert set(expected.columns) == set(
        data.columns
    ), f"Expected: {expected.columns}\nGot: {data.columns}"

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.equals(
        expected
    ), f"Expected: {expected}\nGot: {ordered_columns}"

    await store.upsert_into(
        FeatureLocation.feature_view("loan"),
        pl.DataFrame(
            {
                "loan_amount": [4000],
                "loan_id": [10000],
                "event_timestamp": [datetime.now(tz=timezone.utc)],
                "personal_income": [59000],
                "loan_status": [1],
            }
        ),
    )
    await store.upsert_into(
        FeatureLocation.feature_view("loan"),
        pl.DataFrame(
            {
                "loan_id": [10001],
                "loan_amount": [4000],
                "event_timestamp": [datetime.now(tz=timezone.utc)],
                "personal_income": [59000],
                "loan_status": [1],
            }
        ),
    )
    await store.feature_view("loan").all().to_polars()


@pytest.mark.asyncio
async def test_delta_overwrite():
    from polars.testing import assert_frame_equal
    from aligned import FileSource
    from aligned.request.retrieval_request import RetrievalRequest

    dir = FileSource.directory("test_data/temp")
    source = dir.delta_at("delta/upsert")

    request = RetrievalRequest(
        name="test",
        location=FeatureLocation.feature_view("test"),
        entities={Feature("e", FeatureType.int8())},
        features={
            Feature("a", FeatureType.int8()),
            Feature("b", FeatureType.int8()),
            Feature("x", FeatureType.int8()),
        },
        derived_features=set(),
    )
    initial_data = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [1, 2, 3, 1, 2, 3],
            "e": [1, 2, 3, 4, 5, 6],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    new_data = pl.DataFrame(
        {"a": [1, 1, 1, 1], "b": [2, 2, 3, 1], "e": [1, 2, 3, 7], "x": [7, 8, 9, 10]}
    )
    expected = pl.concat([new_data, initial_data]).unique(["e"], keep="first")

    await source.delete()
    await source.overwrite(
        RetrievalJob.from_polars_df(initial_data, [request]), request
    )
    data = await source.to_polars()
    assert_frame_equal(data, initial_data, check_column_order=False)

    await source.upsert(RetrievalJob.from_polars_df(new_data, [request]), request)
    new = await source.to_polars()
    assert_frame_equal(new.sort("e"), expected.sort("e"), check_column_order=False)


@pytest.mark.asyncio
async def test_delta_overwrite_predicate():
    from polars.testing import assert_frame_equal
    from aligned import FileSource
    from aligned.request.retrieval_request import RetrievalRequest

    dir = FileSource.directory("test_data/temp")
    source = dir.delta_at("delta/upsert")

    request = RetrievalRequest(
        name="test",
        location=FeatureLocation.feature_view("test"),
        entities={Feature("e", FeatureType.int8())},
        features={
            Feature("a", FeatureType.int8()),
            Feature("b", FeatureType.int8()),
            Feature("x", FeatureType.int8()),
        },
        derived_features=set(),
    )
    initial_data = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [1, 2, 3, 1, 2, 3],
            "e": [1, 2, 3, 4, 5, 6],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    new_data = pl.DataFrame(
        {"a": [1, 1, 1, 1], "b": [2, 2, 3, 1], "e": [1, 2, 3, 7], "x": [7, 8, 9, 10]}
    )
    expected = pl.concat([initial_data.filter(pl.col("a") != 1), new_data])

    # clean up the data
    await source.delete()
    await source.overwrite(
        RetrievalJob.from_polars_df(initial_data, [request]), request
    )
    data = await source.to_polars()
    assert_frame_equal(data, initial_data, check_column_order=False)

    await source.overwrite(
        RetrievalJob.from_polars_df(new_data, [request]),
        request,
        predicate=Expression.from_value(pl.col("a") == 1),
    )
    new = await source.to_polars()
    assert_frame_equal(new.sort("e"), expected.sort("e"), check_column_order=False)


@pytest.mark.asyncio
async def test_delta_overwrite_predicate_with_partition():
    from polars.testing import assert_frame_equal
    from aligned import FileSource
    from aligned.request.retrieval_request import RetrievalRequest

    dir = FileSource.directory("test_data/temp")
    source = dir.delta_at(
        "delta/overwrite-with-partition", config=DeltaConfig(partition_by="a")
    )

    request = RetrievalRequest(
        name="test",
        location=FeatureLocation.feature_view("test"),
        entities={Feature("e", FeatureType.int8())},
        features={
            Feature("a", FeatureType.int8()),
            Feature("b", FeatureType.int8()),
            Feature("x", FeatureType.int8()),
        },
        derived_features=set(),
    )
    initial_data = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [1, 2, 3, 1, 2, 3],
            "e": [1, 2, 3, 4, 5, 6],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    new_data = pl.DataFrame(
        {"a": [1, 1, 1, 1], "b": [2, 2, 3, 1], "e": [1, 2, 3, 7], "x": [7, 8, 9, 10]}
    )
    expected = pl.concat([initial_data.filter(pl.col("a") != 1), new_data])

    # clean up the data
    await source.delete()
    await source.overwrite(
        RetrievalJob.from_polars_df(initial_data, [request]), request
    )
    data = await source.to_polars()
    assert_frame_equal(data.sort("e"), initial_data.sort("e"), check_column_order=False)

    await source.overwrite(
        RetrievalJob.from_polars_df(new_data, [request]),
        request,
        predicate=Expression.from_value(pl.col("a") == 1),
    )
    new = await source.to_polars()
    assert_frame_equal(new.sort("e"), expected.sort("e"), check_column_order=False)


@pytest.mark.asyncio
async def test_parquet(point_in_time_data_test: DataTest) -> None:
    store = ContractStore.experimental()

    for source in point_in_time_data_test.sources:
        view = source.view
        view_name = view.metadata.name

        file_source = FileSource.delta_at(f"test_data/temp/delta/{view_name}")
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
        point_in_time_data_test.feature_reference,
        event_timestamp_column="event_timestamp",
    )
    data = (await job.to_lazy_polars()).collect()

    expected = point_in_time_data_test.expected_output

    assert (
        expected.shape == data.shape
    ), f"Expected: {expected.shape}\nGot: {data.shape}"
    assert set(expected.columns) == set(
        data.columns
    ), f"Expected: {expected.columns}\nGot: {data.columns}"

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.equals(
        expected
    ), f"Expected: {expected}\nGot: {ordered_columns}"


@pytest.mark.asyncio
async def test_parquet_without_event_timestamp(
    point_in_time_data_test_wituout_event_timestamp: DataTest,
) -> None:
    store = ContractStore.experimental()

    for source in point_in_time_data_test_wituout_event_timestamp.sources:
        view = source.view
        view_name = view.metadata.name

        file_source = FileSource.delta_at(f"test_data/temp/delta/{view_name}")
        await file_source.delete()
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

    assert (
        expected.shape == data.shape
    ), f"Expected: {expected.shape}\nGot: {data.shape}"
    assert set(expected.columns) == set(
        data.columns
    ), f"Expected: {expected.columns}\nGot: {data.columns}"

    ordered_columns = data.select(expected.columns)
    assert ordered_columns.equals(
        expected
    ), f"Expected: {expected}\nGot: {ordered_columns}"


@pytest.mark.asyncio
async def test_read_csv(point_in_time_data_test: DataTest) -> None:
    store = ContractStore.experimental()

    for source in point_in_time_data_test.sources:
        view = source.view
        view_name = view.metadata.name
        if "_agg" in view_name:
            continue

        file_source = FileSource.delta_at(
            f"test_data/delta/{view_name}",
            date_formatter=DateFormatter.unix_timestamp(),
        )
        await file_source.delete()

        view.metadata = FeatureView.metadata_with(  # type: ignore
            name=view.metadata.name,
            description=view.metadata.description,
            batch_source=file_source,
        )
        compiled = view.compile()
        assert isinstance(compiled.source, DeltaFileSource)
        assert compiled.source.path == file_source.path

        store.add_compiled_view(compiled)

        Path(file_source.path.as_posix()).unlink(missing_ok=True)

        await store.feature_view(compiled.name).insert(
            store.feature_view(compiled.name).process_input(source.data)
        )

        # Polars
        stored = await store.feature_view(compiled.name).all().to_polars()
        df = stored.select(source.data.columns)
        assert df.equals(source.data)


@pytest.mark.asyncio
async def test_read_optional_csv() -> None:
    source = FileSource.delta_at("test_data/temp/delta/optional_test")
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "c": [1, 2, 3],
        }
    )
    await source.delete()
    await source.write_polars(df.lazy())

    @feature_view(name="test", source=source)
    class Test:
        a = Int32().as_entity()
        b = Int32().is_optional()
        c = Int32()

        filled = b.fill_na(0)

    expected_df = df.with_columns(pl.lit(None).alias("b"), pl.lit(0).alias("filled"))
    loaded = await Test.query().all().to_polars()  # type: ignore

    assert loaded.equals(expected_df.select(loaded.columns))

    facts = await Test.query().features_for({"a": [2]}).to_polars()  # type: ignore
    assert expected_df.filter(pl.col("a") == 2).equals(
        facts.select(expected_df.columns)
    )


@pytest.mark.asyncio
async def test_read_optional_view() -> None:
    source = FileSource.delta_at("test_data/temp/delta/optional_test")
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "c": [1, 2, 3],
        }
    )
    await source.delete()
    await source.write_polars(df.lazy())

    @feature_view(name="test_a", source=source)
    class TestA:
        a = Int32().as_entity()
        c = Int32()

    @feature_view(name="test", source=TestA)  # type: ignore
    class Test:
        a = Int32().as_entity()
        b = Int32().is_optional()
        c = Int32()

        filled = b.fill_na(0)

    expected_df = df.with_columns(pl.lit(None).alias("b"), pl.lit(0).alias("filled"))
    loaded = await Test.query().all().to_polars()  # type: ignore

    assert loaded.equals(expected_df.select(loaded.columns))

    facts = await Test.query().features_for({"a": [2]}).to_polars()  # type: ignore
    assert expected_df.filter(pl.col("a") == 2).equals(
        facts.select(expected_df.columns)
    )
