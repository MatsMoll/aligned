from datetime import datetime
from aladdin.feature import Feature
from aladdin.derivied_feature import DerivedFeature
from dataclasses import dataclass

@dataclass
class RetrivalRequest:

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
        return [feature.name for feature in self.all_required_features]

    @property
    def all_features(self) -> set[Feature]:
        return self.features.union(self.derived_features)

    @property
    def all_feature_names(self) -> set[str]:
        return [feature.name for feature in self.all_features]

    @property
    def entity_names(self) -> set[str]:
        return {entity.name for entity in self.entities}
