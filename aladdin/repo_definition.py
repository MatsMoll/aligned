from pathlib import Path
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.feature_view.combined_view import CompiledCombinedFeatureView
from aladdin.codable import Codable
from dataclasses import dataclass

from aladdin.online_source import OnlineSource


@dataclass
class RepoDefinition(Codable):
    feature_views: set[CompiledFeatureView]
    combined_feature_views: set[CompiledCombinedFeatureView]
    models: dict[str, list[str]]
    online_source: OnlineSource


    @staticmethod
    def from_url(url: str) -> "RepoDefinition":
        if url.startswith("s3:"):
            raise NotImplementedError()
        
        path = Path(url)
        if not path.is_file():
            raise ValueError(f"Path is not a local file: {url}")
        
        return RepoDefinition.from_json(path.read_text())