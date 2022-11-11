from dataclasses import dataclass, field

from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import EventTimestamp, Feature


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
                if dep_ref.is_derived:
                    dep_features = [feat for feat in self.derived_features if feat.name == dep_ref.name]
                    if not dep_features:
                        raise ValueError(
                            'Unable to find the referenced feature. This is most likely a bug in the systemd'
                        )
                    dep_feature = dep_features[0]
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
                    features=features - self.entities,
                    derived_features=derived_features,
                    event_timestamp=self.event_timestamp,
                )
            ],
        )

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        entites = '\n'.join([str(entity) for entity in self.entities])
        input_features = '\n'.join([str(features) for features in self.features])
        transformed_features = '\n'.join([str(features) for features in self.derived_features])
        return f"""
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


@dataclass
class CompiledCombinedFeatureView(Codable):
    name: str
    features: set[DerivedFeature]  # FIXME: Should combine this and feature_referances into one class.
    feature_referances: dict[str, list[RetrivalRequest]]

    @property
    def entity_features(self) -> set[Feature]:
        values = set()
        for requests in self.feature_referances.values():
            for request in requests:
                values.update(request.entities)
        return values

    @property
    def entity_names(self) -> set[str]:
        return {feature.name for feature in self.entity_features}

    @property
    def request_all(self) -> FeatureRequest:
        requests: dict[str, RetrivalRequest] = {}
        entities = set()
        for sub_requests in self.feature_referances.values():
            for request in sub_requests:
                entities.update(request.entities)
                if request.feature_view_name not in requests:
                    requests[request.feature_view_name] = RetrivalRequest(
                        feature_view_name=request.feature_view_name,
                        entities=request.entities,
                        features=set(),
                        derived_features=set(),
                        event_timestamp=request.event_timestamp,
                    )
                requests[request.feature_view_name].derived_features.update(request.derived_features)
                requests[request.feature_view_name].features.update(request.features)
                requests[request.feature_view_name].entities.update(request.entities)

        requests[self.name] = RetrivalRequest(
            feature_view_name=self.name,
            entities=entities,
            features=set(),
            derived_features=self.features,
            event_timestamp=None,
        )

        return FeatureRequest(
            self.name,
            features_to_include={feature.name for feature in self.features.union(entities)},
            needed_requests=RetrivalRequest.combine(list(requests.values())),
        )

    def requests_for(self, feature_names: set[str]) -> FeatureRequest:
        entities = self.entity_names
        dependent_views: dict[str, RetrivalRequest] = {}
        for feature in feature_names:
            if feature in entities:
                continue

            if feature not in self.feature_referances.keys():
                raise ValueError(f'Invalid feature {feature} in {self.name}')

            requests = self.feature_referances[feature]
            for request in requests:
                if request.feature_view_name not in dependent_views:
                    dependent_views[request.feature_view_name] = RetrivalRequest(
                        feature_view_name=request.feature_view_name,
                        entities=request.entities,
                        features=set(),
                        derived_features=set(),
                        event_timestamp=request.event_timestamp,
                    )
                current = dependent_views[request.feature_view_name]
                current.derived_features = current.derived_features.union(request.derived_features)
                current.features = current.features.union(request.features)
                dependent_views[request.feature_view_name] = current

        dependent_views[self.name] = RetrivalRequest(  # Add the request we want
            feature_view_name=self.name,
            entities=self.entity_features,
            features=set(),
            derived_features=[feature for feature in self.features if feature.name in feature_names],
            event_timestamp=None,
        )

        return FeatureRequest(
            self.name,
            features_to_include=feature_names,
            needed_requests=list(dependent_views.values()),
        )

    def __hash__(self) -> int:
        return hash(self.name)
