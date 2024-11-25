from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, TypeVar
from uuid import uuid4
from datetime import datetime, timezone

from mashumaro.types import SerializableType
from aligned.data_source.batch_data_source import CodableBatchDataSource
from aligned.request.retrival_request import RequestResult

from aligned.sources.local import Deletable, StorageFileSource
from aligned.schemas.codable import Codable

T = TypeVar('T')


class DatasetStorageFactory:

    supported_stores: dict[str, type[DatasetStore]] = {}

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


class DatasetMetadataInterface(Protocol):
    id: str
    created_at: datetime
    content: RequestResult
    name: str | None
    description: str | None
    tags: list[str] | None


@dataclass
class DatasetMetadata(Codable):
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str | None = field(default=None)
    description: str | None = field(default=None)
    tags: list[str] | None = field(default=None)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SingleDatasetMetadata(Codable):
    source: CodableBatchDataSource
    content: RequestResult
    created_at: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str | None = field(default=None)
    description: str | None = field(default=None)
    tags: list[str] | None = field(default=None)

    def format_as_job(self, job: T) -> T:
        from aligned.retrival_job import RetrivalJob, SupervisedJob

        if isinstance(job, RetrivalJob):
            return self.source.all(job.request_result)
        elif isinstance(job, SupervisedJob):
            return SupervisedJob(  # type: ignore
                job=self.source.all(job.request_result),
                target_columns=job.target_columns,
                should_filter_out_null_targets=job.should_filter_out_null_targets,
            )
        else:
            raise NotImplementedError(f"Can't convert {type(self)} to type {type(job)} job.")


@dataclass
class TrainDatasetMetadata(Codable, DatasetMetadataInterface):

    name: str | None

    content: RequestResult

    train_dataset: CodableBatchDataSource
    test_dataset: CodableBatchDataSource

    validation_dataset: CodableBatchDataSource | None = field(default=None)

    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    train_size_fraction: float | None = field(default=None)
    test_size_fraction: float | None = field(default=None)
    validate_size_fraction: float | None = field(default=None)

    target: list[str] | None = field(default=None)

    description: str | None = field(default=None)
    tags: list[str] | None = field(default=None)

    @property
    def as_datasets(self) -> list[SingleDatasetMetadata]:
        sources = [
            (['train'], self.train_dataset),
            (['test'], self.test_dataset),
        ]
        if self.validation_dataset:
            sources.append((['validation'], self.validation_dataset))

        datasets = []
        for tags, source in sources:
            datasets.append(
                SingleDatasetMetadata(
                    source=source,
                    content=self.content,
                    created_at=self.created_at,
                    id=self.id,
                    name=self.name,
                    description=self.description,
                    tags=tags + (self.tags or []),
                )
            )
        return datasets


@dataclass
class GroupedDatasetList(Codable):

    raw_data: list[SingleDatasetMetadata]

    train_test: list[TrainDatasetMetadata]
    train_test_validation: list[TrainDatasetMetadata]

    @property
    def all_datasets(self) -> list[SingleDatasetMetadata]:
        datasets = self.raw_data.copy()
        for train_test in self.train_test:
            datasets.extend(train_test.as_datasets)
        for train_test in self.train_test_validation:
            datasets.extend(train_test.as_datasets)
        return datasets

    @property
    def all(self) -> Sequence[DatasetMetadataInterface]:
        return self.raw_data + self.train_test + self.train_test_validation


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

    async def store_dataset(self, metadata: SingleDatasetMetadata) -> None:
        raise NotImplementedError(type(self))

    async def store_train_test(self, metadata: TrainDatasetMetadata) -> None:
        raise NotImplementedError(type(self))

    async def store_train_test_validate(self, metadata: TrainDatasetMetadata) -> None:
        raise NotImplementedError(type(self))

    async def metadata_for(self, dataset_id: str) -> DatasetMetadataInterface | None:
        raise NotImplementedError(type(self))

    async def datasets_with_tag(self, tag: str) -> list[SingleDatasetMetadata]:
        raise NotImplementedError(type(self))

    async def latest_dataset_with_tag(self, tag: str) -> DatasetMetadataInterface | None:
        raise NotImplementedError(type(self))

    async def delete_metadata_for(self, dataset_id: str) -> DatasetMetadataInterface | None:
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
            )

    def index_of(self, metadata_id: str, array: Sequence[DatasetMetadataInterface]) -> int | None:

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

    async def store_dataset(self, metadata: SingleDatasetMetadata) -> None:
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

    async def datasets_with_tag(self, tag: str) -> list[SingleDatasetMetadata]:
        datasets = await self.list_datasets()
        return [dataset for dataset in datasets.all_datasets if dataset.tags and tag in dataset.tags]

    async def latest_dataset_with_tag(self, tag: str) -> DatasetMetadataInterface | None:
        datasets = await self.list_datasets()
        latest_dataset: DatasetMetadataInterface | None = None

        for dataset in datasets.all:
            if not (dataset.tags and tag in dataset.tags):
                continue

            if (latest_dataset is None) or (dataset.created_at > latest_dataset.created_at):
                latest_dataset = dataset

        return latest_dataset

    async def metadata_for(self, dataset_id: str) -> DatasetMetadataInterface | None:
        datasets = await self.list_datasets()
        for dataset in datasets.all:
            if dataset.id == dataset_id:
                return dataset
        return None

    async def delete_metadata_for(self, dataset_id: str) -> DatasetMetadataInterface | None:
        datasets = await self.list_datasets()

        async def delete_dataset(source: CodableBatchDataSource):
            if isinstance(source, Deletable):
                await source.delete()

        index = self.index_of(dataset_id, datasets.raw_data)
        if index is not None:
            dataset = datasets.raw_data.pop(index)
            await delete_dataset(dataset.source)
            return dataset

        index = self.index_of(dataset_id, datasets.train_test)
        if index is not None:
            dataset = datasets.train_test.pop(index)
            await delete_dataset(dataset.train_dataset)
            await delete_dataset(dataset.test_dataset)
            return dataset

        index = self.index_of(dataset_id, datasets.train_test_validation)
        if index is not None:
            dataset = datasets.train_test_validation.pop(index)
            await delete_dataset(dataset.train_dataset)
            await delete_dataset(dataset.test_dataset)
            if dataset.validation_dataset:
                await delete_dataset(dataset.validation_dataset)
            return dataset

        return None
