import pytest

from aligned.feature_store import FeatureStore
from aligned.retrival_job import split
from aligned.sources.local import CsvFileSource


@pytest.mark.asyncio
async def test_split(scan_with_datetime: CsvFileSource) -> None:

    data_set_size = 10
    end_ratio = 0.8
    result_size = data_set_size * end_ratio

    dataset = await scan_with_datetime.enricher().as_df()
    subset = dataset[:data_set_size]

    split_set = split(subset, event_timestamp_column='created_at', start_ratio=0, end_ratio=end_ratio)
    other_set = split(subset, event_timestamp_column='created_at', start_ratio=end_ratio, end_ratio=1)

    assert split_set.shape[0] == result_size
    assert other_set.shape[0] == (data_set_size - result_size)


@pytest.mark.asyncio
async def test_train_test_validate_set(titanic_feature_store: FeatureStore) -> None:

    dataset_size = 100
    train_fraction = 0.6
    validation_fraction = 0.2

    train_size = int(round(dataset_size * train_fraction))
    test_size = int(round(dataset_size * (1 - train_fraction - validation_fraction)))
    validate_size = int(round(dataset_size * validation_fraction))

    dataset = (
        await titanic_feature_store.feature_view('titanic')
        .all(limit=dataset_size)
        .train_set(train_fraction, target_column='survived')
        .validation_set(validation_fraction)
        .to_pandas()
    )

    assert dataset.train.data.shape[0] == train_size
    assert dataset.test.data.shape[0] == test_size
    assert dataset.validate.data.shape[0] == validate_size

    assert 'passenger_id' in dataset.data.columns
    assert 'survived' in dataset.data.columns

    assert 'passenger_id' not in dataset.train_input.columns
    assert 'survived' not in dataset.train_input.columns
