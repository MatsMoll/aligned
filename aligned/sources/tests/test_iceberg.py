from datetime import timezone
import pytest
import polars as pl

from aligned import (
    ContractStore,
    Int32,
    data_contract,
    String,
)
from aligned.feature_view.feature_view import FeatureView
from aligned.retrieval_job import RetrievalJob
from aligned.sources.iceberg import IcebergCatalog
from conftest import DataTest


@pytest.mark.asyncio
async def test_read_iceberg(point_in_time_data_test: DataTest) -> None:
    store = ContractStore.experimental()

    for source in point_in_time_data_test.sources:
        view = source.view
        view_name = view.metadata.name
        if "_agg" in view_name:
            continue

        file_source = IcebergCatalog().table("test")

        req = source.view.request

        await file_source.delete()
        await file_source.overwrite(
            job=RetrievalJob.from_convertable(source.data, [req]).derive_features(),
            request=req,
        )

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
async def test_upsert_iceberg(point_in_time_data_test: DataTest) -> None:
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

        file_source = IcebergCatalog().table(view_name)

        req = view.request
        unique_data = source.data.lazy().unique(req.entity_names)

        await file_source.delete()
        await file_source.overwrite(
            RetrievalJob.from_convertable(unique_data, [req]).derive_features(), req
        )

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

    await store.data_contract("loan").upsert(
        pl.DataFrame(
            {
                "loan_amount": [4000],
                "loan_id": [10000],
                "event_timestamp": [datetime.now(tz=timezone.utc)],
                "personal_income": [59000],
                "loan_status": [1],
            }
        )
    )
    await store.data_contract("loan").upsert(
        pl.DataFrame(
            {
                "loan_id": [10001],
                "loan_amount": [4000],
                "event_timestamp": [datetime.now(tz=timezone.utc)],
                "personal_income": [59000],
                "loan_status": [1],
            }
        )
    )
    upserted = (
        await store.data_contract("loan").features_for({"loan_id": [10001]}).to_polars()
    ).to_dicts()[0]

    assert upserted["loan_amount"] == 4000
    assert upserted["personal_income"] == 59000
    assert upserted["loan_status"] == 1


@data_contract(source=IcebergCatalog().table("test_delete"))
class TestTable:
    x = Int32().as_entity()

    value = String()
    other = Int32()


@pytest.mark.asyncio
async def test_iceberg_partial_delete():
    table = TestTable.query()

    await table.delete()
    df = await table.all().to_polars()
    assert df.height == 0

    await table.overwrite(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "value": ["a", "b", "a", "b", "a", "b"],
            "other": [1, 1, 1, 2, 2, 2],
        }
    )

    df = await table.all().to_polars()
    assert df.height == 6
    await table.delete(TestTable().other == 1)

    df = await table.all().to_polars()
    assert df.height == 3


@pytest.mark.asyncio
async def test_iceberg_partial_overwrite():
    table = TestTable.query()

    await table.delete()
    df = await table.all().to_polars()
    assert df.height == 0

    await table.overwrite(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "value": ["a", "b", "a", "b", "a", "b"],
            "other": [1, 1, 1, 2, 2, 2],
        }
    )

    df = await table.all().to_polars()
    assert df.height == 6

    await table.overwrite(
        {"x": [1, 2, 3, 4], "value": ["a", "b", "a", "b"], "other": [1, 2, 3, 4]},
        predicate=TestTable().other == 1,
    )

    df = await table.all().to_polars()
    assert df.height == 7

    await table.overwrite(
        {"x": [1, 2, 3, 4], "value": ["a", "b", "a", "b"], "other": [1, 2, 3, 4]}
    )

    df = await table.all().to_polars()
    assert df.height == 4
