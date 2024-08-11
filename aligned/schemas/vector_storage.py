from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pyarrow as pa
from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable
from aligned.schemas.feature import Feature, FeatureLocation, FeatureType


class VectorStorageFactory:

    supported_storages: dict[str, type[VectorStorage]]

    _shared: VectorStorageFactory | None = None

    def __init__(self) -> None:
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

    async def n_similar_polars(self, vector: pl.Series, n: int) -> pl.DataFrame:
        raise NotImplementedError()


def pyarrow_schema(features: list[Feature]) -> pa.Schema:
    def pa_dtype(dtype: FeatureType) -> pa.DataType:
        pa_types = {
            'int8': pa.int8(),
            'int16': pa.int16(),
            'int32': pa.int32(),
            'int64': pa.int64(),
            'uint8': pa.uint8(),
            'uint16': pa.uint16(),
            'uint32': pa.uint32(),
            'uint64': pa.uint64(),
            'float': pa.float64(),
            'double': pa.float64(),
            'string': pa.large_string(),
            'uuid': pa.large_string(),
            'date': pa.date64(),
            'embedding': pa.large_list(pa.float32()),
            'datetime': pa.float64(),
            'list': pa.large_list(pa.int32()),
            'array': pa.large_list(pa.int32()),
            'bool': pa.bool_(),
        }

        if dtype.name in pa_types:
            return pa_types[dtype.name]

        if dtype.is_datetime:
            return pa.timestamp('us', tz=dtype.datetime_timezone)

        if dtype.is_embedding:
            embedding_size = dtype.embedding_size()
            if embedding_size:
                return pa.list_(pa.float32(), embedding_size)
            return pa_types['embedding']

        if dtype.is_array:
            array_sub_dtype = dtype.array_subtype()
            if array_sub_dtype:
                return pa.large_list(pa_dtype(array_sub_dtype))

            return pa.large_list(pa.string())

        raise ValueError(f"Unsupported dtype: {dtype}")

    def pa_field(feature: Feature) -> pa.Field:
        from aligned.schemas.constraints import Optional

        is_nullable = Optional() in (feature.constraints or set())

        pa_type = pa_dtype(feature.dtype)
        return pa.field(feature.name, pa_type, nullable=is_nullable)

    return pa.schema([pa_field(feature) for feature in features])


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

    def pyarrow_schema(self) -> pa.Schema:
        features = [self.vector]

        if self.metadata:
            features.extend(self.metadata)
        if self.entities:
            features.extend(self.entities)

        sorted_features = sorted(features, key=lambda feature: feature.name)

        return pyarrow_schema(sorted_features)
