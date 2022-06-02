from dataclasses import dataclass

from aladdin.codable import Codable
from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import EventTimestamp, Feature
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest


@dataclass
class CompiledFeatureView(Codable):
    name: str
    description: str
    tags: dict[str, str]
    batch_data_source: BatchDataSource
    # stream_data_source: StreamDataSource | None

    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]
    event_timestamp: EventTimestamp

    @property
    def full_schema(self) -> set[Feature]:
        return self.entities.union(self.features).union(self.derived_features)

    @property
    def entitiy_names(self) -> set[str]:
        return {entity.name for entity in self.entities}

    @property
    def request_all(self) -> FeatureRequest:
        return FeatureRequest(
            self.name,
            {feature.name for feature in self.full_schema},
            needed_requests=[
                RetrivalRequest(
                    feature_view_name=self.name,
                    entities=self.entities,
                    features=self.features,
                    derived_features=self.derived_features,
                    event_timestamp=self.event_timestamp,
                )
            ],
        )

    def request_for(self, feature_names: set[str]) -> FeatureRequest:

        features = {feature for feature in self.features if feature.name in feature_names}.union(
            self.entities
        )
        derived_features = {feature for feature in self.derived_features if feature.name in feature_names}

        def dependent_features_for(
            feature: DerivedFeature,
        ) -> tuple[set[Feature], set[Feature]]:
            core_features = set()
            intermediate_features = set()

            for dep_ref in feature.depending_on:
                if dep_ref.is_derivied:
                    dep_feature = [feat for feat in self.derived_features if feat.name == dep_ref.name][0]
                    intermediate_features.add(dep_feature)
                    core, intermediate = dependent_features_for(dep_feature)
                    features.update(core)
                    intermediate_features.update(intermediate)
                else:
                    dep_feature = [
                        feat for feat in self.features.union(self.entities) if feat.name == dep_ref.name
                    ][0]
                    core_features.add(dep_feature)

            return core_features, intermediate_features

        for dep_feature in derived_features.copy():
            core, intermediate = dependent_features_for(dep_feature)
            features.update(core)
            derived_features.update(intermediate)

        return FeatureRequest(
            self.name,
            feature_names,
            needed_requests=[
                RetrivalRequest(
                    feature_view_name=self.name,
                    entities=self.entities,
                    features=features,
                    derived_features=derived_features,
                    event_timestamp=self.event_timestamp,
                )
            ],
        )

    def __hash__(self) -> int:
        return hash(self.name)
