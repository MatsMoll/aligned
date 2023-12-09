import pytest
from aligned import feature_view, Int32, FileSource
import polars as pl


@feature_view(name='left', source=FileSource.csv_at('some_file.csv'))
class LeftData:

    some_id = Int32().as_entity()

    feature = Int32()


@feature_view(name='right', source=FileSource.csv_at('some_file.csv'))
class RightData:

    some_id = Int32().as_entity()

    other_feature = Int32()


@pytest.mark.asyncio
async def test_join_different_types_polars() -> None:

    left_data = LeftData.from_data(  # type: ignore
        pl.DataFrame(
            {'some_id': [1, 2, 3], 'feature': [2, 3, 4]}, schema={'some_id': pl.Int8, 'feature': pl.Int32}
        )
    )

    right_data = RightData.from_data(  # type: ignore
        pl.DataFrame(
            {'some_id': [1, 3, 2], 'other_feature': [3, 4, 5]},
            schema={'some_id': pl.Int16, 'other_feature': pl.Int32},
        )
    )

    expected_df = pl.DataFrame(
        data={'some_id': [1, 2, 3], 'feature': [2, 3, 4], 'other_feature': [3, 5, 4]},
        schema={
            'some_id': pl.Int32,
            'feature': pl.Int32,
            'other_feature': pl.Int32,
        },
    )

    new_data = left_data.join(right_data, 'inner', left_on='some_id', right_on='some_id')
    result = await new_data.to_polars()

    joined = result.collect().sort('some_id', descending=False)
    assert joined.frame_equal(expected_df.select(joined.columns))


@pytest.mark.asyncio
async def test_unique_entities() -> None:

    left_data = LeftData.from_data(  # type: ignore
        pl.DataFrame(
            {'some_id': [1, 3, 3], 'feature': [2, 3, 4]}, schema={'some_id': pl.Int8, 'feature': pl.Int32}
        )
    )

    left_data.unique_entities()
