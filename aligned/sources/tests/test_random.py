from __future__ import annotations

import pytest

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
