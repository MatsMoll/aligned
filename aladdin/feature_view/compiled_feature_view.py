from dataclasses import dataclass, field

from aladdin.codable import Codable

# from aladdin.codable import Codable
from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.data_source.stream_data_source import StreamDataSource
from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import EventTimestamp, Feature
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest

# from typing import Generic, Optional, TypeVar


# class VersionableData(Codable):
#     valid_from: datetime

# VersionData = TypeVar("VersionData", bound=VersionableData)

# @dataclass
# class VersionedData(Generic[VersionData], Codable):

#     identifier: str
#     versions: list[VersionData]

#     def __init__(self, identifier: str, versions: list[VersionData]) -> None:
#         self.identifier = identifier
#         self.versions = versions

#     @property
#     def latest(self) -> VersionData:
#         return self.versions[0]

#     def version_valid_at(self, timestamp: datetime) -> VersionData | None:
#         for version in self.versions:
#             if version.valid_from < timestamp:
#                 return version
#         return None

#     def __hash__(self) -> int:
#         return hash(self.identifier)


@dataclass
class CompiledFeatureView(Codable):
    name: str
    description: str
    tags: dict[str, str]
    batch_data_source: BatchDataSource

    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]
    event_timestamp: EventTimestamp | None = field(default=None)
    stream_data_source: StreamDataSource | None = field(default=None)

    # valid_from: datetime

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

    # def version_at(self, timestamp: datetime) -> Optional["CompiledFeatureView"]:
    #     if self.created_at < timestamp:
    #         return self
    #     if prev_version := self.prev_version:
    #         return prev_version.version_at(timestamp)
    #     else:
    #         return None

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        entites = '\n'.join([str(entity) for entity in self.entities])
        input_features = '\n'.join([str(features) for features in self.features])
        transformed_features = '\n'.join([str(features) for features in self.derived_features])
        string_representation = f"""
{self.name}
Description: {self.description}
Tags: {self.tags}

Entities:
{entites}

Event Timestamp:
{self.event_timestamp}

Input features:
{input_features}

Transformed features:
{transformed_features}
        """
        return string_representation

    # def __eq__(self, other: object) -> bool:

    #     if not isinstance(other, CompiledFeatureView):
    #         return False

    #     feature_difference = (other.features.union(self.features) -
    # other.features.intersection(self.features))
    #     if feature_difference:
    #         return False

    #     derived_feature_difference = other.derived_features.union(
    #         self.derived_features
    #     ) - other.derived_features.intersection(self.derived_features)
    #     if derived_feature_difference:
    #         return False

    #     entity_differance = other.entities.union(self.entities) - other.entities.intersection(self.entities)
    #     if entity_differance:
    #         return False

    #     if self.event_timestamp != other.event_timestamp:
    #         return False

    #     return True
