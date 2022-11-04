from dataclasses import dataclass, field

from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import EventTimestamp, Feature


@dataclass
class RetrivalRequest(Codable):
    """
    Describes all the information needed for a request to be successful.

    This do not mean all the data is shown to the end user,
    as there may be some features that depend on other features.
    """

    feature_view_name: str
    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]
    event_timestamp: EventTimestamp | None = field(default=None)

    @property
    def feature_names(self) -> list[str]:
        return [feature.name for feature in self.features]

    @property
    def request_result(self) -> 'RequestResult':
        return RequestResult.from_request(self)

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

        for feature in features:
            depth = feature.depth
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

    @staticmethod
    def unsafe_combine(requests: list['RetrivalRequest']) -> list['RetrivalRequest']:

        result_request = RetrivalRequest(
            feature_view_name='random',
            entities=set(),
            features=set(),
            derived_features=set(),
            event_timestamp=None,
        )
        for request in requests:
            result_request.derived_features.update(request.derived_features)
            result_request.features.update(request.features)
            result_request.entities.update(request.entities)

        return result_request


@dataclass
class RequestResult(Codable):
    """
    Describes the returend response of a request
    """

    entities: set[Feature]
    features: set[Feature]
    event_timestamp: str | None

    def __add__(self, obj: 'RequestResult') -> 'RequestResult':
        return RequestResult(
            entities=self.entities.union(obj.entities),
            features=self.features.union(obj.features),
            event_timestamp='event_timestamp' if self.event_timestamp or obj.event_timestamp else None,
        )

    def filter_features(self, features_to_include: set[str]) -> 'RequestResult':
        return RequestResult(
            entities=self.entities,
            features={feature for feature in self.features if feature.name in features_to_include},
            event_timestamp=self.event_timestamp,
        )

    @staticmethod
    def from_request(request: RetrivalRequest) -> 'RequestResult':
        return RequestResult(
            entities=request.entities,
            features=request.all_features - request.entities,
            event_timestamp=request.event_timestamp.name if request.event_timestamp else None,
        )

    @staticmethod
    def from_request_list(requests: list[RetrivalRequest]) -> 'RequestResult':
        request_len = len(requests)
        if request_len == 0:
            return RequestResult(entities=set(), features=set(), event_timestamp=None)
        elif request_len > 1:
            return RequestResult(
                entities=set().union(*[request.entities for request in requests]),
                features=set().union(*[request.all_features - request.entities for request in requests]),
                event_timestamp='event_timestamp'
                if any(request.event_timestamp for request in requests)
                else None,
            )
        else:
            return RequestResult.from_request(requests[0])

    @staticmethod
    def from_result_list(requests: list['RequestResult']) -> 'RequestResult':
        request_len = len(requests)
        if request_len == 0:
            return RequestResult(entities=set(), features=set(), event_timestamp=None)
        elif request_len > 1:
            return RequestResult(
                entities=set().union(*[request.entities for request in requests]),
                features=set().union(*[request.features for request in requests]),
                event_timestamp='event_timestamp'
                if any(request.event_timestamp for request in requests)
                else None,
            )
        else:
            return requests[0]


@dataclass
class FeatureRequest(Codable):
    """Representing a request of a set of features
    This dataclass would be used to represent which
    features to fetch for a given model.

    It would therefore contain the different features that is for the endgoal.
    But also which features that is needed in order to get the wanted features.

    E.g:
    Let's say we have the following feature view:

    ```
    class TitanicPassenger(FeatureView):

        ... # The metadata and entity is unrelevent here

        sex = String()
        is_male, is_female = sex.one_hot_encode(["male", "female"])
    ```

    If we ask for only the feature `is_male` the this dataclass would contain

    name = the name of the feature view it originates from, or some something random
    features_to_include = {'is_male'}
    needed_requests = [
        features={'sex'}, # would fetch only the sex feature, as `is_male` relies on it
        derived_features={'is_male'} # The feature to be computed
    ]
    """

    name: str
    features_to_include: set[str]
    needed_requests: list[RetrivalRequest]

    @property
    def needs_event_timestamp(self) -> bool:
        return any(request.event_timestamp for request in self.needed_requests)

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.needed_requests)
