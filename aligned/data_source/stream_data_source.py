from abc import ABC
from dataclasses import dataclass, field
from typing import Optional

from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable


class StreamDataSourceFactory:

    supported_data_sources: dict[str, type['StreamDataSource']]

    _shared: Optional['StreamDataSourceFactory'] = None

    def __init__(self) -> None:
        from aligned.redis.config import RedisStreamSource

        self.supported_data_sources = {
            HttpStreamSource.name: HttpStreamSource,
            RedisStreamSource.name: RedisStreamSource,
        }

    @classmethod
    def shared(cls) -> 'StreamDataSourceFactory':
        if cls._shared:
            return cls._shared
        cls._shared = StreamDataSourceFactory()
        return cls._shared


class StreamDataSource(ABC, Codable, SerializableType):
    """
    Used to determend if an API call should be created, or if we should listen to a stream.
    """

    name: str
    topic_name: str

    def _serialize(self) -> dict:
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> 'StreamDataSource':
        name = value['name']
        if name not in StreamDataSourceFactory.shared().supported_data_sources:
            raise ValueError(
                f"Unknown stream data source id: '{name}'.\nRemember to add the"
                ' data source to the StreamDataSourceFactory.supported_data_sources if'
                ' it is a custom type.'
            )
        del value['name']
        data_class = StreamDataSourceFactory.shared().supported_data_sources[name]
        return data_class.from_dict(value)


@dataclass
class HttpStreamSource(StreamDataSource):

    topic_name: str
    mappings: dict[str, str] = field(default_factory=dict)

    name: str = 'http'

    def map_values(self, mappings: dict[str, str]) -> 'HttpStreamSource':
        return HttpStreamSource(topic_name=self.topic_name, mappings=self.mappings | mappings)
