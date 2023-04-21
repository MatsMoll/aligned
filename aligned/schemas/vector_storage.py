from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable
from aligned.schemas.feature import Feature, FeatureLocation


class VectorStorageFactory:

    supported_storages: dict[str, type[VectorStorage]]

    _shared: VectorStorageFactory | None = None

    def __init__(self):
        self.supported_storages = {}

        for storage in VectorStorage.__subclasses__():
            self.supported_storages[storage.type_name] = storage

    @classmethod
    def shared(cls) -> VectorStorageFactory:
        if cls._shared is None:
            cls._shared = VectorStorageFactory()
        return cls._shared


class VectorStorage(Codable, SerializableType):

    type_name: str

    def _serialize(self) -> dict:
        assert (
            self.type_name in VectorStorageFactory.shared().supported_storages
        ), f'VectorStorage {self.type_name} is not supported'
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> VectorStorage:
        name = value['type_name']
        if name not in VectorStorageFactory.shared().supported_storages:
            raise ValueError(f'VectorStorage {name} is not supported')
        del value['type_name']
        return VectorStorageFactory.shared().supported_storages[name].from_dict(value)

    async def create_index(self, index: VectorIndex) -> None:
        raise NotImplementedError()

    async def upsert_polars(self, df: pl.LazyFrame, index: VectorIndex) -> None:
        raise NotImplementedError()


@dataclass
class VectorIndex(Codable):

    location: FeatureLocation
    vector: Feature
    vector_dim: int
    metadata: list[Feature]
    storage: VectorStorage
    entities: list[Feature]

    def __pre_serialize__(self) -> VectorIndex:
        assert isinstance(self.vector_dim, int), f'got {self.vector_dim}, expected int'
        assert isinstance(self.storage, VectorStorage)
        assert isinstance(self.location, FeatureLocation)
        assert isinstance(self.vector, Feature)
        assert isinstance(self.metadata, list), f'metadata must be a list, got {type(self.metadata)}'
        assert isinstance(self.entities, list), f'entities must be a list, got {type(self.entities)}'

        return self
