import pytest

from pathlib import Path
from aligned.feature_store import FeatureStore
from aligned.retrival_job import split
from aligned.schemas.folder import DatasetMetadata
from aligned.sources.local import CsvFileSource, FileSource


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


@pytest.mark.asyncio
async def test_train_test_validate_set_new(titanic_feature_store: FeatureStore) -> None:
    from aligned.schemas.folder import JsonDatasetStore

    unlink_paths = [
        'test_data/titanic-sets.json',
        'test_data/titanic-train.csv',
        'test_data/titanic-test.csv',
        'test_data/titanic-validate.csv',
    ]

    for path_str in unlink_paths:
        path = Path(path_str)
        if path.exists():
            path.unlink()

    dataset_size = 100
    train_fraction = 0.6
    validation_fraction = 0.2

    train_size = int(round(dataset_size * train_fraction))
    test_size = int(round(dataset_size * (1 - train_fraction - validation_fraction)))
    validate_size = int(round(dataset_size * validation_fraction))

    dataset_store = FileSource.json_at('test_data/titanic-sets.json')
    dataset = (
        titanic_feature_store.feature_view('titanic')
        .all(limit=dataset_size)
        .train_test_validate(train_fraction, validation_fraction, target_column='survived')
        .store_dataset(
            dataset_store,
            metadata=DatasetMetadata(
                id='titanic_test',
            ),
            train_source=FileSource.csv_at('test_data/titanic-train.csv'),
            test_source=FileSource.csv_at('test_data/titanic-test.csv'),
            validate_source=FileSource.csv_at('test_data/titanic-validate.csv'),
        )
    )

    train = await dataset.train.to_pandas()
    test = await dataset.test.to_pandas()
    validate = await dataset.validate.to_pandas()

    store = JsonDatasetStore(dataset_store)
    datasets = await store.list_datasets()

    assert store.to_json() != None

    assert len(datasets.train_test_validation) == 1
    train_dataset = datasets.train_test_validation[0]

    assert train.data.shape[0] == train_size
    assert test.data.shape[0] == test_size
    assert validate.data.shape[0] == validate_size

    assert train_dataset.train_size_fraction == train_fraction

    assert 'passenger_id' in train.data.columns
    assert 'survived' in train.data.columns

    assert 'passenger_id' not in train.input.columns
    assert 'survived' not in train.input.columns
