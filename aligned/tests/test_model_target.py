from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from aligned import FeatureStore, model_contract, String, Int32
from aligned.schemas.feature import FeatureLocation


@pytest.mark.asyncio
async def test_titanic_model_with_targets(titanic_feature_store: FeatureStore) -> None:

    entity_list = [1, 4, 5, 6, 7, 30, 31, 2]

    dataset = (
        await titanic_feature_store.model('titanic')
        .with_labels()
        .features_for({'passenger_id': entity_list})
        .to_pandas()
    )

    assert dataset.input.shape == (8, 5)
    assert dataset.labels.shape == (8, 1)
    assert dataset.entities.shape == (8, 1)

    assert np.all(dataset.entities['passenger_id'].to_numpy() == entity_list)


@pytest.mark.asyncio
async def test_titanic_model_with_targets_and_scd(titanic_feature_store_scd: FeatureStore) -> None:

    entities = pl.DataFrame(
        {
            'passenger_id': [1, 2, 3, 4, 5, 6, 7],
            'event_timestamp': [
                datetime(2023, 2, 6, tzinfo=timezone.utc),
                datetime(2023, 2, 6, tzinfo=timezone.utc),
                datetime(2023, 1, 2, tzinfo=timezone.utc),
                datetime(2023, 2, 6, tzinfo=timezone.utc),
                datetime(2023, 2, 6, tzinfo=timezone.utc),
                datetime(2023, 2, 6, tzinfo=timezone.utc),
                datetime(2023, 2, 6, tzinfo=timezone.utc),
            ],
        }
    )
    expected_data = pl.DataFrame(
        {
            'survived': [True, False, True, False, True, True, True],
            'is_male': [True, False, True, True, False, True, True],
            'age': [22, 50, 70, 14, 44, 72, 22],
        }
    )

    dataset = (
        await titanic_feature_store_scd.model('titanic')
        .with_labels()
        .features_for(entities.to_dict(as_series=False), event_timestamp_column='event_timestamp')
        .to_lazy_polars()
    )

    input_df = dataset.input.collect()
    target_df = dataset.labels.collect()

    assert target_df['survived'].equals(expected_data['survived'])
    assert input_df['is_male'].equals(expected_data['is_male'])
    assert input_df['age'].equals(expected_data['age'])


@pytest.mark.asyncio
async def test_model_wrapper() -> None:
    from aligned.compiler.model import ModelContractWrapper

    @model_contract(
        name='test_model',
        features=[],
    )
    class TestModel:
        id = Int32().as_entity()

        a = Int32()

    test_model_features = TestModel()

    @model_contract(name='new_model', features=[test_model_features.a])
    class NewModel:

        id = Int32().as_entity()

        x = String()

    model_wrapper: ModelContractWrapper = NewModel
    compiled = model_wrapper.compile()
    assert len(compiled.features.default_features) == 1

    feature = list(compiled.features.default_features)[0]

    assert feature.location == FeatureLocation.model('test_model')
    assert feature.name == 'a'


def test_with_renames() -> None:
    from aligned import FileSource

    source = FileSource.parquet_at('test_data/test_model.parquet').with_renames({'some_id': 'id'})
    other = source.with_renames({'other_id': 'id'})

    assert source.mapping_keys == {'some_id': 'id'}
    assert other.mapping_keys == {'other_id': 'id'}


@pytest.mark.asyncio
async def test_model_insert_predictions() -> None:
    """
    Test the insert (aka. ish append) method on the feature store.
    """
    from aligned import FileSource, FeatureStore

    path = 'test_data/test_model.parquet'

    @model_contract(
        name='test_model',
        features=[],
        prediction_source=FileSource.parquet_at(path).with_renames({'some_id': 'id'}),
    )
    class TestModel:
        id = Int32().as_entity()

        a = Int32()

    store = FeatureStore.experimental()
    initial_frame = pl.DataFrame({'id': [1, 2, 3], 'a': [1, 2, 3]})
    initial_frame.write_parquet(path)

    expected_frame = pl.DataFrame({'id': [1, 2, 3, 1, 2, 3], 'a': [10, 14, 20, 1, 2, 3]})

    store.add_compiled_model(TestModel.compile())  # type: ignore

    await store.insert_into(FeatureLocation.model('test_model'), {'id': [1, 2, 3], 'a': [10, 14, 20]})

    preds = await store.model('test_model').all_predictions().to_polars()

    stored_data = pl.read_parquet(path).select(id=pl.col('some_id'), a=pl.col('a'))
    assert stored_data.equals(expected_frame)

    assert preds.select(expected_frame.columns).equals(expected_frame)


@pytest.mark.asyncio
async def test_model_insert_predictions_csv() -> None:
    """
    Test the insert (aka. ish append) method on the feature store.
    """
    from aligned import FileSource, FeatureStore

    path = 'test_data/test_model.csv'

    @model_contract(
        name='test_model',
        features=[],
        prediction_source=FileSource.csv_at(path).with_renames({'some_id': 'id'}),
    )
    class TestModel:
        id = Int32().as_entity()

        a = Int32()

    store = FeatureStore.experimental()

    initial_frame = pl.DataFrame({'some_id': [1, 2, 3], 'a': [1, 2, 3]})
    initial_frame.write_csv(path)

    expected_frame = pl.DataFrame({'id': [1, 2, 3, 1, 2, 3], 'a': [10, 14, 20, 1, 2, 3]})

    store.add_compiled_model(TestModel.compile())  # type: ignore

    await store.insert_into(FeatureLocation.model('test_model'), {'id': [1, 2, 3], 'a': [10, 14, 20]})

    preds = await store.model('test_model').all_predictions().log_each_job().to_polars()

    stored_data = pl.read_csv(path).select(id=pl.col('some_id'), a=pl.col('a'))
    assert stored_data.equals(expected_frame)

    assert preds.select(expected_frame.columns).equals(expected_frame)


@pytest.mark.asyncio
async def test_model_upsert_predictions() -> None:
    """
    Test the insert (aka. ish append) method on the feature store.
    """
    from aligned import FileSource, FeatureStore

    path = 'test_data/test_model.parquet'

    @model_contract(name='test_model', features=[], prediction_source=FileSource.parquet_at(path))
    class TestModel:
        id = Int32().as_entity()

        a = Int32()

    store = FeatureStore.experimental()
    initial_frame = pl.DataFrame({'id': [1, 2, 3, 4], 'a': [1, 2, 3, 4]})
    initial_frame.write_parquet(path)

    expected_frame = pl.DataFrame({'id': [1, 2, 3, 4], 'a': [10, 14, 20, 4]})

    store.add_compiled_model(TestModel.compile())  # type: ignore

    await store.upsert_into(FeatureLocation.model('test_model'), {'id': [1, 2, 3], 'a': [10, 14, 20]})

    stored_data = pl.read_parquet(path).sort('id')

    columns = set(stored_data.columns).difference(expected_frame.columns)
    assert len(columns) == 0
    assert stored_data.select(expected_frame.columns).equals(expected_frame)
