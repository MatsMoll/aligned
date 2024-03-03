from __future__ import annotations

from dataclasses import dataclass, field

from mashumaro.types import SerializableType
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.request.retrival_request import RequestResult

from aligned.sources.local import StorageFileSource
from aligned.schemas.codable import Codable


class DatasetStorageFactory:

    supported_stores: dict[str, type[DatasetStore]] = dict()

    _shared: DatasetStorageFactory | None = None

    def __init__(self) -> None:
        dataset_types = [
            JsonDatasetStore,
        ]
        for dataset_type in dataset_types:
            self.supported_stores[dataset_type.name] = dataset_type

    @classmethod
    def shared(cls) -> DatasetStorageFactory:
        if cls._shared:
            return cls._shared
        cls._shared = DatasetStorageFactory()
        return cls._shared


@dataclass
class DatasetMetadata(Codable):

    id: str
    name: str | None = field(default=None)
    description: str | None = field(default=None)
    tags: list[str] | None = field(default=None)


@dataclass
class TrainDatasetMetadata(Codable):

    id: str
    name: str | None

    request_result: RequestResult

    train_dataset: BatchDataSource
    test_dataset: BatchDataSource

    validation_dataset: BatchDataSource | None = field(default=None)

    train_size_fraction: float | None = field(default=None)
    test_size_fraction: float | None = field(default=None)
    validate_size_fraction: float | None = field(default=None)

    target: list[str] | None = field(default=None)

    description: str | None = field(default=None)
    tags: list[str] | None = field(default=None)


@dataclass
class GroupedDatasetList(Codable):

    raw_data: list[DatasetMetadata]

    train_test: list[TrainDatasetMetadata]
    train_test_validation: list[TrainDatasetMetadata]

    active_learning: list[DatasetMetadata]

    @property
    def all(self) -> list[DatasetMetadata]:
        return self.raw_data + self.train_test + self.train_test_validation + self.active_learning


class DatasetStore(Codable, SerializableType):

    name: str

    def _serialize(self) -> dict:
        assert self.name in DatasetStorageFactory.shared().supported_stores, f'Unknown type_name: {self.name}'
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> DatasetStore:
        name_type = value['name']
        if name_type not in DatasetStorageFactory.shared().supported_stores:
            supported = DatasetStorageFactory.shared().supported_stores.keys()
            raise ValueError(
                f"Unknown batch data source id: '{name_type}'.\nRemember to add the"
                ' data source to the FolderFactory.supported_folders if'
                ' it is a custom type.'
                f' Have access to the following types: {supported}'
            )
        del value['name']
        data_class = DatasetStorageFactory.shared().supported_stores[name_type]
        return data_class.from_dict(value)

    async def list_datasets(self) -> GroupedDatasetList:
        raise NotImplementedError(type(self))

    async def store_raw_data(self, metadata: DatasetMetadata) -> None:
        raise NotImplementedError(type(self))

    async def store_train_test(self, metadata: TrainDatasetMetadata) -> None:
        raise NotImplementedError(type(self))

    async def store_train_test_validate(self, metadata: TrainDatasetMetadata) -> None:
        raise NotImplementedError(type(self))

    async def store_active_learning(self, metadata: DatasetMetadata) -> None:
        raise NotImplementedError(type(self))

    async def metadata_for(self, dataset_id: str) -> DatasetMetadata | None:
        raise NotImplementedError(type(self))

    async def delete_metadata_for(self, dataset_id: str) -> DatasetMetadata | None:
        raise NotImplementedError(type(self))


@dataclass
class JsonDatasetStore(DatasetStore):

    source: StorageFileSource
    name = 'json'

    async def list_datasets(self) -> GroupedDatasetList:
        try:
            data = await self.source.read()
            return GroupedDatasetList.from_json(data)
        except FileNotFoundError:
            return GroupedDatasetList(
                raw_data=[],
                train_test=[],
                train_test_validation=[],
                active_learning=[],
            )

    def index_of(
        self, metadata_id: str, array: list[DatasetMetadata] | list[TrainDatasetMetadata]
    ) -> int | None:

        for i, dataset in enumerate(array):
            if dataset.id == metadata_id:
                return i
        return None

    async def store_train_test(self, metadata: TrainDatasetMetadata) -> None:
        datasets = await self.list_datasets()

        index = self.index_of(metadata.id, datasets.train_test)
        if index is None:
            datasets.train_test.append(metadata)
        else:
            datasets.train_test[index] = metadata

        data = datasets.to_json()
        if isinstance(data, str):
            data = data.encode('utf-8')
        await self.source.write(data)

    async def store_train_test_validate(self, metadata: TrainDatasetMetadata) -> None:
        datasets = await self.list_datasets()

        index = self.index_of(metadata.id, datasets.train_test_validation)
        if index is None:
            datasets.train_test_validation.append(metadata)
        else:
            datasets.train_test_validation[index] = metadata

        data = datasets.to_json()
        if isinstance(data, str):
            data = data.encode('utf-8')

        await self.source.write(data)

    async def store_raw_data(self, metadata: DatasetMetadata) -> None:
        datasets = await self.list_datasets()

        index = self.index_of(metadata.id, datasets.raw_data)
        if index is not None:
            datasets.raw_data[index] = metadata
        else:
            datasets.raw_data.append(metadata)

        data = datasets.to_json()
        if isinstance(data, str):
            data = data.encode('utf-8')
        await self.source.write(data)

    async def store_active_learning(self, metadata: DatasetMetadata) -> None:
        datasets = await self.list_datasets()

        index = self.index_of(metadata.id, datasets.active_learning)
        if index is None:
            datasets.active_learning.append(metadata)
        else:
            datasets.active_learning[index] = metadata

        data = datasets.to_json()
        if isinstance(data, str):
            data = data.encode('utf-8')
        await self.source.write(data)

    async def metadata_for(self, dataset_id: str) -> DatasetMetadata | None:
        datasets = await self.list_datasets()
        for dataset in datasets.all:
            if dataset.id == dataset_id:
                return dataset
        return None

    async def delete_metadata_for(self, dataset_id: str) -> DatasetMetadata | None:
        datasets = await self.list_datasets()
        index = self.index_of(dataset_id, datasets.all)
        if index is None:
            return None
        return datasets.all[index]
