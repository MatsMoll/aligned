from __future__ import annotations

import pytest

from aligned import feature_view, String, Int16, List
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


@pytest.mark.asyncio
async def test_random_derived_feature() -> None:
    @feature_view(source=RandomDataSource())
    class Features:
        some_id = Int16().as_entity()
        values = List(String())
        contains_a = values.contains("a")

        integer = Int16()

        is_above_10 = integer > 10

    query = Features.query()

    random_values = await query.all(limit=10).to_polars()
    assert random_values.height == 10

    with_inject = await query.features_for(
        {"values": [["A", "b"], ["a", "B"]], "some_id": [1, 2]}
    ).to_polars()
    assert with_inject[Features().contains_a.name].to_list() == [False, True]
