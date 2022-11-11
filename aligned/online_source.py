import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from mashumaro.types import SerializableType

from aligned.feature_source import FeatureSource, InMemoryFeatureSource
from aligned.schemas.codable import Codable
from aligned.schemas.feature_view import CompiledFeatureView

logger = logging.getLogger(__name__)


class OnlineSourceFactory:

    supported_sources: dict[str, type['OnlineSource']]

    _shared: Optional['OnlineSourceFactory'] = None

    def __init__(self) -> None:
        from aligned.redis.config import RedisOnlineSource

        self.supported_sources = {
            BatchOnlineSource.source_type: BatchOnlineSource,
            RedisOnlineSource.source_type: RedisOnlineSource,
        }

    @classmethod
    def shared(cls) -> 'OnlineSourceFactory':
        if cls._shared:
            return cls._shared
        cls._shared = OnlineSourceFactory()
        return cls._shared


class OnlineSource(ABC, Codable, SerializableType):
    """
    A codable source, that can create a feature source.

    This is sepearted form the FeatureSource, as this may contain additional
    information that should not be decoded.
    """

    source_type: str

    @abstractmethod
    def feature_source(self, feature_views: set[CompiledFeatureView]) -> FeatureSource:
        pass

    def _serialize(self) -> dict:
        return self.to_dict()

    def __hash__(self) -> int:
        return hash(self.source_type)

    @classmethod
    def _deserialize(cls, value: dict[str, Any]) -> 'OnlineSource':
        try:
            name_type = value['source_type']

            if name_type not in OnlineSourceFactory.shared().supported_sources:
                raise ValueError(
                    f"Unknown online source id: '{name_type}'.\nRemember to add the"
                    " data source to the 'OnlineSourceFactory.supported_sources' if it"
                    ' is a custom type.'
                )
            del value['source_type']

            data_class = OnlineSourceFactory.shared().supported_sources[name_type]

            return data_class.from_dict(value)
        except Exception as error:
            logger.error(error)
            raise error


@dataclass
class BatchOnlineSource(OnlineSource):
    """
    Setup the feature source based on the feature store information.
    This is needed in order to make the queries more efficent.
    """

    source_type = 'batch'

    def feature_source(self, feature_views: set[CompiledFeatureView]) -> FeatureSource:
        from aligned.feature_source import BatchFeatureSource

        return BatchFeatureSource(
            sources={fv.name: fv.batch_data_source for fv in feature_views},
        )


@dataclass
class InMemoryOnlineSource(OnlineSource):

    source_type: str = 'in-mem'

    def feature_source(self, feature_views: set[CompiledFeatureView]) -> FeatureSource:
        return InMemoryFeatureSource({})
