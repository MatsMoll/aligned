from aladdin.codable import Codable
from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.feature import Feature
from aladdin.derivied_feature import DerivedFeature
from dataclasses import dataclass

@dataclass
class RetrivalRequest(Codable):

    feature_view_name: str
    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]

    @property
    def feature_names(self) -> list[str]:
        return [feature.name for feature in self.features]

    @property
    def all_required_features(self) -> set[Feature]:
        depends_on = set()
        for feature in self.derived_features:
            depends_on = depends_on.union(feature.depending_on)
        return self.features.union(depends_on)

    @property
    def all_required_feature_names(self) -> set[str]:
        return {feature.name for feature in self.all_required_features}

    @property
    def all_features(self) -> set[Feature]:
        return self.features.union(self.derived_features)

    @property
    def all_feature_names(self) -> set[str]:
        return {feature.name for feature in self.all_features}

    @property
    def entity_names(self) -> set[str]:
        return {entity.name for entity in self.entities}


@dataclass
class FeatureRequest(Codable):
    features: set[str]
    needed_requests: list[RetrivalRequest]

    
    def core_requests_given(self, sources: dict[str, BatchDataSource]) -> dict[BatchDataSource, RetrivalRequest]:
        return {sources[request.feature_view_name]: request for request in self.needed_requests if request.feature_view_name in sources}

    def combined_requests_given(self, sources: dict[str, BatchDataSource]) -> list[RetrivalRequest]:
        return [request for request in self.needed_requests if request.feature_view_name not in sources]