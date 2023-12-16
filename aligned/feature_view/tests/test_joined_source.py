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


@feature_view(name='right', source=FileSource.csv_at('some_file.csv'))
class RightOtherIdData:

    other_id = Int32().as_entity()

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
async def test_join_different_join_keys() -> None:

    left_data = LeftData.from_data(  # type: ignore
        pl.DataFrame(
            {'some_id': [1, 2, 3], 'feature': [2, 3, 4]}, schema={'some_id': pl.Int8, 'feature': pl.Int32}
        )
    )

    right_data = RightOtherIdData.from_data(  # type: ignore
        pl.DataFrame(
            {'other_id': [1, 3, 2], 'other_feature': [3, 4, 5]},
            schema={'other_id': pl.Int16, 'other_feature': pl.Int32},
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

    new_data = left_data.join(right_data, 'inner', left_on='some_id', right_on='other_id')

    result = await new_data.to_polars()
    req_result = new_data.request_result

    joined = result.collect().sort('some_id', descending=False)

    assert joined.frame_equal(expected_df.select(joined.columns))
    assert joined.select(req_result.entity_columns).frame_equal(expected_df.select(['some_id']))


@pytest.mark.asyncio
async def test_unique_entities() -> None:

    data = LeftData.from_data(  # type: ignore
        pl.DataFrame(
            {'some_id': [1, 3, 3], 'feature': [2, 3, 4]}, schema={'some_id': pl.Int8, 'feature': pl.Int32}
        )
    )
    expected_df = pl.DataFrame(
        data={'some_id': [1, 3], 'feature': [2, 4]},
        schema={
            'some_id': pl.Int8,
            'feature': pl.Int32,
        },
    )

    result = await data.unique_on(['some_id'], sort_key='feature').to_polars()
    sorted = result.sort('some_id').select(['some_id', 'feature']).collect()

    assert sorted.frame_equal(expected_df)
