import pytest

from aligned.feature_store import ContractStore
from aligned.schemas.folder import DatasetMetadata
from aligned.sources.local import FileSource


@pytest.mark.asyncio
async def test_train_test_validate_set(titanic_feature_store: ContractStore) -> None:

    dataset_size = 100
    train_fraction = 0.6
    validation_fraction = 0.2

    train_size = int(round(dataset_size * train_fraction))
    test_size = int(round(dataset_size * (1 - train_fraction - validation_fraction)))
    validate_size = int(round(dataset_size * validation_fraction))

    datasets = (
        titanic_feature_store.feature_view('titanic')
        .all(limit=dataset_size)
        .train_test_validate(train_fraction, validation_fraction, target_column='survived')
    )
    train = await datasets.train.to_pandas()
    test = await datasets.test.to_pandas()
    validate = await datasets.validate.to_pandas()

    assert train.data.shape[0] == train_size
    assert test.data.shape[0] == test_size
    assert validate.data.shape[0] == validate_size

    assert 'passenger_id' in train.data.columns
    assert 'survived' in train.data.columns

    assert 'passenger_id' not in train.input.columns
    assert 'survived' not in train.input.columns


@pytest.mark.asyncio
async def test_train_test_validate_set_new(titanic_feature_store: ContractStore) -> None:
    from pathlib import Path
    from aligned.schemas.folder import JsonDatasetStore

    dataset_size = 100
    train_fraction = 0.6
    validation_fraction = 0.2

    train_size = int(round(dataset_size * train_fraction))
    test_size = int(round(dataset_size * (1 - train_fraction - validation_fraction)))
    validate_size = int(round(dataset_size * validation_fraction))

    dataset_store = FileSource.json_at('test_data/temp/titanic-sets.json')
    train_source = FileSource.csv_at('test_data/temp/titanic-train.csv')
    test_source = FileSource.csv_at('test_data/temp/titanic-test.csv')
    validate_source = FileSource.csv_at('test_data/temp/titanic-validate.csv')

    delete_files = [dataset_store.path, train_source.path, test_source.path, validate_source.path]

    for file in delete_files:
        path = Path(file)
        if path.exists():
            path.unlink()

    dataset = await (
        titanic_feature_store.feature_view('titanic')
        .all(limit=dataset_size)
        .train_test_validate(train_fraction, validation_fraction, target_column='survived')
        .store_dataset(
            dataset_store,
            metadata=DatasetMetadata(
                id='titanic_test',
            ),
            train_source=train_source,
            test_source=test_source,
            validate_source=validate_source,
        )
    )

    train = await dataset.train.to_pandas()
    test = await dataset.test.to_pandas()
    validate = await dataset.validate.to_pandas()

    store = JsonDatasetStore(dataset_store)
    datasets = await store.list_datasets()

    assert store.to_json() is not None

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
