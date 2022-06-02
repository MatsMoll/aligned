from dataclasses import dataclass

from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.data_source.stream_data_source import StreamDataSource
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView


@dataclass
class FeatureViewMetadata:
    name: str
    description: str
    tags: dict[str, str]
    batch_source: BatchDataSource
    stream_source: StreamDataSource | None = None

    @staticmethod
    def from_compiled(view: CompiledFeatureView) -> 'FeatureViewMetadata':
        return FeatureViewMetadata(
            name=view.name,
            description=view.description,
            tags=view.tags,
            batch_source=view.batch_data_source,
            stream_source=None,
        )
