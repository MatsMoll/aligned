import pytest
from aligned import feature_view, Int32, CustomMethodDataSource
from aligned.request.retrival_request import RetrivalRequest
import polars as pl


async def all_data(request: RetrivalRequest, limit: int | None = None) -> pl.LazyFrame:

    return pl.DataFrame({'some_id': [1, 2, 3], 'feature': [2, 3, 4]}).lazy()


@feature_view(name='right', source=CustomMethodDataSource.from_methods(all_data=all_data))
class CustomSourceData:

    some_id = Int32().as_entity()

    feature = Int32()


@pytest.mark.asyncio
async def test_custom_source() -> None:

    result = await CustomSourceData.query().all().to_polars()

    assert result.equals(pl.DataFrame({'some_id': [1, 2, 3], 'feature': [2, 3, 4]}).select(result.columns))
