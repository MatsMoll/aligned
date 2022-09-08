from dataclasses import dataclass

from aladdin.codable import Codable
from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import EventTimestamp, Feature


@dataclass
class RetrivalRequest(Codable):

    feature_view_name: str
    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]
    event_timestamp: EventTimestamp | None

    @property
    def feature_names(self) -> list[str]:
        return [feature.name for feature in self.features]

    def derived_feature_map(self) -> dict[str, DerivedFeature]:
        return {feature.name: feature for feature in self.derived_features}

    @property
    def all_required_features(self) -> set[Feature]:
        return self.features - self.entities

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
        feature_orders: list[set] = []
        feature_map = self.derived_feature_map()
        features = self.derived_features
        dependent_features = self.derived_features.copy()

        while dependent_features:
            feature = dependent_features.pop()
            for dep_ref in feature.depending_on:
                if dep_ref.name == feature.name:
                    continue

                if dep := feature_map.get(dep_ref.name):
                    feature_deps[feature.name].add(dep)
                    if dep.name not in feature_deps:
                        dependent_features.add(dep)
                        feature_map[dep.name] = dep

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

    @staticmethod
    def combine(requests: list['RetrivalRequest']) -> list['RetrivalRequest']:
        grouped_requests: dict[str, RetrivalRequest] = {}
        entities = set()
        for request in requests:
            entities.update(request.entities)
            fv_name = request.feature_view_name
            if fv_name not in grouped_requests:
                grouped_requests[fv_name] = RetrivalRequest(
                    feature_view_name=fv_name,
                    entities=request.entities,
                    features=request.features,
                    derived_features=request.derived_features,
                    event_timestamp=request.event_timestamp,
                )
            else:
                grouped_requests[fv_name].derived_features.update(request.derived_features)
                grouped_requests[fv_name].features.update(request.features)
                grouped_requests[fv_name].entities.update(request.entities)

        return list(grouped_requests.values())


@dataclass
class FeatureRequest(Codable):
    name: str
    features_to_include: set[str]
    needed_requests: list[RetrivalRequest]

    @property
    def needs_event_timestamp(self) -> bool:
        return any(request.event_timestamp for request in self.needed_requests)
