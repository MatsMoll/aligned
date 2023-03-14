from datetime import datetime

import numpy as np
import polars as pl
import pytest

from aligned import FeatureStore


@pytest.mark.asyncio
async def test_titanic_model_with_targets(titanic_feature_store: FeatureStore) -> None:

    entity_list = [1, 4, 5, 6, 7, 30, 31, 2]

    dataset = (
        await titanic_feature_store.model('titanic')
        .with_target()
        .features_for({'passenger_id': entity_list})
        .to_pandas()
    )

    assert dataset.input.shape == (8, 5)
    assert dataset.target.shape == (8, 1)
    assert dataset.entities.shape == (8, 1)

    assert np.all(dataset.entities['passenger_id'].to_numpy() == entity_list)


@pytest.mark.asyncio
async def test_titanic_model_with_targets_and_scd(titanic_feature_store_scd: FeatureStore) -> None:

    entities = pl.DataFrame(
        {
            'passenger_id': [1, 2, 3, 4, 5, 6, 7],
            'event_timestamp': [
                datetime(2023, 2, 6),
                datetime(2023, 2, 6),
                datetime(2023, 1, 2),
                datetime(2023, 2, 6),
                datetime(2023, 2, 6),
                datetime(2023, 2, 6),
                datetime(2023, 2, 6),
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
        .with_target()
        .features_for(entities.to_dict())
        .to_polars()
    )

    input_df = dataset.input.collect()
    target_df = dataset.target.collect()

    assert target_df['survived'].series_equal(expected_data['survived'])
    assert input_df['is_male'].series_equal(expected_data['is_male'])
    assert input_df['age'].series_equal(expected_data['age'])
