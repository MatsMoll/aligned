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
        .features_for(entities.to_dict(as_series=False))
        .to_polars()
    )

    input_df = dataset.input.collect()
    target_df = dataset.labels.collect()

    assert target_df['survived'].series_equal(expected_data['survived'])
    assert input_df['is_male'].series_equal(expected_data['is_male'])
    assert input_df['age'].series_equal(expected_data['age'])


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
    assert len(compiled.features) == 1

    feature = list(compiled.features)[0]

    assert feature.location == FeatureLocation.model('test_model')
    assert feature.name == 'a'
