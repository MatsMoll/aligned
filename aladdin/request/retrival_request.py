from aladdin.codable import Codable
from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.feature import Feature, EventTimestamp
from aladdin.derivied_feature import DerivedFeature
from dataclasses import dataclass

@dataclass
class RetrivalRequest(Codable):

    feature_view_name: str
    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]
    event_timestamp: EventTimestamp

    core_features: set[Feature]
    intermediate_features: set[DerivedFeature]


    @property
    def feature_names(self) -> list[str]:
        return [feature.name for feature in self.features]

    def derived_feature_map(self) -> dict[str, DerivedFeature]:
        return {feature.name: feature for feature in self.derived_features.union(self.intermediate_features)}

    @property
    def all_required_features(self) -> set[Feature]:
        return self.core_features

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

    def derived_features_order(self) -> list[set[DerivedFeature]]:
        from collections import defaultdict
        feature_deps = defaultdict(set)
        feature_orders = []
        feature_map = self.derived_feature_map()
        features = self.derived_features.union(self.intermediate_features)

        for feature in features:
            for dep_ref in feature.depending_on:
                if dep := feature_map.get(dep_ref.name):
                    feature_deps[feature.name].add(dep)
        
        def depths(feature: DerivedFeature) -> int:
            if feature.name not in feature_deps or not feature_deps[feature.name]:
                return 0
            max = 0
            for dep in feature.depending_on:
                depth = depths(feature_map[dep.name])
                if depth > max:
                    max = depth

            return max + 1

        for feature in features:
            depth = depths(feature)
            while depth >= len(feature_orders):
                feature_orders.append(set())
            feature_orders[depth].add(feature_map[feature.name])
        
        return feature_orders

            

        



@dataclass
class FeatureRequest(Codable):
    features: set[str]
    needed_requests: list[RetrivalRequest]

    
    def core_requests_given(self, sources: dict[str, BatchDataSource]) -> dict[BatchDataSource, RetrivalRequest]:
        return {sources[request.feature_view_name]: request for request in self.needed_requests if request.feature_view_name in sources}

    def combined_requests_given(self, sources: dict[str, BatchDataSource]) -> list[RetrivalRequest]:
        return [request for request in self.needed_requests if request.feature_view_name not in sources]