import math

import pytest

from aligned.feature_store import FeatureStore
from aligned.local.source import CsvFileSource
from aligned.retrival_job import split


@pytest.mark.asyncio
async def test_split(titanic_source: CsvFileSource) -> None:

    data_set_size = 100
    end_ratio = 0.8
    result_size = data_set_size * end_ratio

    dataset = await titanic_source.enricher().as_df()
    subset = dataset[:data_set_size]

    split_set = split(subset, target_column='Survived', start_ratio=0, end_ratio=end_ratio)
    other_set = split(subset, target_column='Survived', start_ratio=end_ratio, end_ratio=1)

    assert split_set.shape[0] == result_size
    assert other_set.shape[0] == (data_set_size - result_size)


@pytest.mark.asyncio
async def test_train_test_validate_set(titanic_feature_store: FeatureStore) -> None:

    dataset_size = 100
    test_fraction = 0.2
    validation_fraction = 0.2

    train_size = int(round(dataset_size * (1 - test_fraction - validation_fraction)))
    test_size = int(round(dataset_size * test_fraction))
    validate_size = int(round(dataset_size * validation_fraction))

    dataset = (
        await titanic_feature_store.feature_view('titanic')
        .all(limit=dataset_size)
        .test_size(test_size=test_fraction, target_column='survived')
        .validation_size(validation_fraction)
        .use_df()
    )

    assert dataset.train.data.shape[0] == train_size
    assert dataset.test.data.shape[0] == test_size
    assert dataset.validate.data.shape[0] == validate_size

    assert 'passenger_id' in dataset.data.columns
    assert 'survived' in dataset.data.columns

    assert 'passenger_id' not in dataset.train_input.columns
    assert 'survived' not in dataset.train_input.columns

    expected_ratio = dataset.output.sum() / dataset.output.shape[0]
    precision = 10
    lower_ratio = math.floor(expected_ratio * precision) / precision
    upper_ratio = math.ceil(expected_ratio * precision) / precision
    train_ratio = dataset.train_output.sum() / dataset.train_output.shape[0]
    test_ratio = dataset.test_output.sum() / dataset.test_output.shape[0]
    validate_ratio = dataset.validate_output.sum() / dataset.validate_output.shape[0]

    assert lower_ratio <= train_ratio <= upper_ratio
    assert lower_ratio <= test_ratio <= upper_ratio
    assert lower_ratio <= validate_ratio <= upper_ratio
