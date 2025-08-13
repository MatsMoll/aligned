from __future__ import annotations

import pytest

from aligned import data_contract, String, Int16, List
from aligned.sources.random_source import RandomDataSource
from conftest import DataTest


@pytest.mark.asyncio
async def test_random_source(point_in_time_data_test: DataTest) -> None:
    source = RandomDataSource()
    view = point_in_time_data_test.sources[0]

    request = view.view.compile().request_all.needed_requests[0]

    limit = 100
    df = await source.all_data(request, limit=limit).to_polars()

    assert df.height == limit

    column = view.data.columns[0]
    await source.write_polars(view.data.select(column).lazy())
    df = await source.all_data(request).to_polars()
    assert df.select(column).equals(view.data.select(column))
    assert len(df.columns) == len(request.all_returned_columns)


@data_contract(source=RandomDataSource())
class Features:
    some_id = Int16().as_entity()
    values = List(String())
    contains_a = values.contains("a")

    integer = Int16()

    is_above_10 = integer > 10


@pytest.mark.asyncio
async def test_random_derived_feature() -> None:
    query = Features.query()

    random_values = await query.all().limit(10).to_polars()
    assert random_values.height == 10

    with_inject = await query.features_for(
        {"values": [["A", "b"], ["a", "B"]], "some_id": [1, 2]}
    ).to_polars()
    assert with_inject[Features().contains_a.name].to_list() == [False, True]


@pytest.mark.asyncio
async def test_random_source_features_for():
    store = Features.query().store
    store = store.update_source_for(
        Features,
        RandomDataSource.with_values(
            {
                "some_id": [1, 2, 3],
                "values": [["A", "b"], ["a", "B"], ["B", "C"]],
            }
        ),
    )

    df = await store.contract(Features).all().to_polars()
    assert df.height == 3
    assert df["contains_a"].to_list() == [False, True, False]

    df = (
        await store.contract(Features)
        .features_for({"integer": [1, 2, 11, 12]})
        .to_polars()
    )

    assert df.height == 4
    assert df["is_above_10"].to_list() == [False, False, True, True]
    assert "values" in df.columns
    assert "some_id" in df.columns


@data_contract(
    source=RandomDataSource.with_values(
        {
            "some_id": [1, 2, 3],
            "values": [["A", "b"], ["a", "B"], ["B", "C"]],
        }
    )
)
class FeaturesWithoutEntities:
    some_id = Int16()
    values = List(String())
    contains_a = values.contains("a")

    integer = Int16()

    is_above_10 = integer > 10


@pytest.mark.asyncio
async def test_random_source_without_entities():
    df = await FeaturesWithoutEntities.query().all().to_polars()
    assert df.height == 3
    assert df["contains_a"].to_list() == [False, True, False]
    assert "is_above_10" in df.columns

    df = (
        await FeaturesWithoutEntities.query()
        .features_for({"integer": [1, 2, 11, 12]})
        .to_polars()
    )

    assert df.height == 4
    assert df["is_above_10"].to_list() == [False, False, True, True]
    assert "values" in df.columns
    assert "some_id" in df.columns
