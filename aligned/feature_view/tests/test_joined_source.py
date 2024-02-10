import pytest
from aligned import feature_view, Int32, FileSource, model_contract
import polars as pl


@feature_view(name='left', source=FileSource.csv_at('test_data/test.csv'))
class LeftData:

    some_id = Int32().as_entity()

    feature = Int32()


@feature_view(name='right', source=FileSource.csv_at('test_data/other.csv'))
class RightData:

    some_id = Int32().as_entity()

    other_feature = Int32()


@feature_view(name='right', source=FileSource.csv_at('test_data/other.csv'))
class RightOtherIdData:

    other_id = Int32().as_entity()

    other_feature = Int32()


@model_contract(
    name='some_model',
    features=[],
    prediction_source=FileSource.csv_at('test_data/other.csv'),
)
class ModelData:

    some_id = Int32().as_entity()

    other_feature = Int32()


model_data = ModelData()


@feature_view(name='joined', source=LeftData.join(model_data, model_data.some_id))
class JoinedData:

    some_id = Int32().as_entity()

    feature = Int32()

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
    result = await new_data.to_lazy_polars()

    joined = result.collect().sort('some_id', descending=False)
    assert joined.equals(expected_df.select(joined.columns))


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

    result = await new_data.to_lazy_polars()
    req_result = new_data.request_result

    joined = result.collect().sort('some_id', descending=False)

    assert joined.equals(expected_df.select(joined.columns))
    assert joined.select(req_result.entity_columns).equals(expected_df.select(['some_id']))


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

    result = await data.unique_on(['some_id'], sort_key='feature').to_lazy_polars()
    sorted = result.sort('some_id').select(['some_id', 'feature']).collect()

    assert sorted.equals(expected_df)


@pytest.mark.asyncio
async def test_load_model_join() -> None:
    df = await JoinedData.query().all().to_pandas()
    assert df.shape == (2, 3)
