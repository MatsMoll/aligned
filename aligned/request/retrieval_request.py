from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pyarrow as pa
import polars as pl

from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import (
    AggregatedFeature,
    AggregateOver,
    DerivedFeature,
)
from aligned.schemas.feature import EventTimestamp, Feature, FeatureLocation

if TYPE_CHECKING:
    from pyspark.sql.types import StructType


@dataclass
class EventTimestampRequest(Codable):
    event_timestamp: EventTimestamp
    entity_column: str | None = field(default=None)


@dataclass
class RetrievalRequest(Codable):
    """
    Describes all the information needed for a request to be successful.

    This do not mean all the data is shown to the end user,
    as there may be some features that depend on other features.
    """

    name: str
    location: FeatureLocation
    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]
    aggregated_features: set[AggregatedFeature] = field(default_factory=set)
    event_timestamp_request: EventTimestampRequest | None = field(default=None)
    features_to_include: set[str] = field(default_factory=set)

    @property
    def event_timestamp(self) -> EventTimestamp | None:
        return (
            self.event_timestamp_request.event_timestamp
            if self.event_timestamp_request
            else None
        )

    def __init__(
        self,
        name: str,
        location: FeatureLocation,
        entities: set[Feature],
        features: set[Feature],
        derived_features: set[DerivedFeature],
        aggregated_features: set[AggregatedFeature] | None = None,
        event_timestamp: EventTimestamp | None = None,
        entity_timestamp_columns: str | None = None,
        event_timestamp_request: EventTimestampRequest | None = None,
        features_to_include: set[str] | None = None,
    ):
        self.name = name
        self.location = location
        self.entities = entities
        self.features = features
        self.derived_features = derived_features
        self.aggregated_features = aggregated_features or set()
        if event_timestamp_request:
            self.event_timestamp_request = event_timestamp_request
        elif event_timestamp:
            self.event_timestamp_request = EventTimestampRequest(
                event_timestamp=event_timestamp, entity_column=entity_timestamp_columns
            )
        self.features_to_include = features_to_include or self.all_feature_names

    def filter_features(self, feature_names: set[str]) -> "RetrievalRequest":
        return RetrievalRequest(
            name=self.name,
            location=self.location,
            entities=self.entities,
            features=self.features,
            derived_features=self.derived_features,
            aggregated_features=self.aggregated_features,
            event_timestamp=self.event_timestamp,
            features_to_include=feature_names,
        )

    @property
    def all_returned_features(self) -> list[Feature]:
        result = self.entities

        if self.event_timestamp and (
            all(
                agg.aggregate_over.window is not None
                for agg in self.aggregated_features
            )
            or len(self.aggregated_features) == 0
        ):
            result = result.union({self.event_timestamp.as_feature()})

        if self.aggregated_features:
            agg_features = [feat.derived_feature for feat in self.aggregated_features]
            agg_names = list(self.aggregated_features)
            derived_after_aggs: set[Feature] = set()
            derived_features = {der.name: der for der in self.derived_features}

            def is_dependent_on_agg_feature(feature: DerivedFeature) -> bool:
                for dep in feature.depending_on_names:
                    if dep in agg_names:
                        return True

                for dep in feature.depending_on_names:
                    if dep in derived_features and is_dependent_on_agg_feature(
                        derived_features[dep]
                    ):
                        return True

                return False

            for feat in self.derived_features:
                if is_dependent_on_agg_feature(feat):
                    derived_after_aggs.add(feat)

            return agg_features + list(derived_after_aggs) + list(result)

        return list(result.union(self.all_features))

    @property
    def intermediate_columns(self) -> list[str]:
        return [
            feat.name
            for feat in self.derived_features
            if feat.name not in self.features_to_include
        ]

    @property
    def all_returned_columns(self) -> list[str]:
        return sorted([feature.name for feature in self.all_returned_features])

    @property
    def returned_features(self) -> set[Feature]:
        return {
            feature
            for feature in self.all_features
            if feature.name in self.features_to_include
        }

    @property
    def feature_names(self) -> list[str]:
        return [feature.name for feature in self.features]

    @property
    def request_result(self) -> "RequestResult":
        return RequestResult.from_request(self)

    def derived_feature_map(self) -> dict[str, DerivedFeature]:
        return {
            feature.name: feature
            for feature in self.derived_features.union(self.derived_aggregated_features)
        }

    @property
    def derived_aggregated_features(self) -> set[DerivedFeature]:
        return {feature.derived_feature for feature in self.aggregated_features}

    @property
    def all_required_features(self) -> set[Feature]:
        return self.features - self.entities

    @property
    def all_required_feature_names(self) -> set[str]:
        return {feature.name for feature in self.all_required_features}

    @property
    def all_features(self) -> set[Feature]:
        return self.features.union(
            {feat for feat in self.derived_features if not feat.name.isnumeric()}
        ).union(
            {
                feature.derived_feature
                for feature in self.aggregated_features
                if not feature.name.isnumeric()
            }
        )

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

    def pyarrow_schema(self) -> pa.Schema:
        from aligned.schemas.vector_storage import pyarrow_schema

        sorted_features = sorted(
            self.all_returned_features, key=lambda feature: feature.name
        )
        return pyarrow_schema(sorted_features)

    def polars_schema(self) -> dict[str, pl.DataType]:
        sorted_features = sorted(
            self.all_returned_features, key=lambda feature: feature.name
        )
        return {feat.name: feat.dtype.polars_type for feat in sorted_features}

    def spark_schema(self) -> "StructType":
        from pyspark.sql.types import StructField, StructType

        return StructType(
            [
                StructField(name=feature.name, dataType=feature.dtype.spark_type)
                for feature in sorted(
                    self.all_returned_features, key=lambda feat: feat.name
                )
            ]
        )

    def aggregate_over(self) -> dict[AggregateOver, set[AggregatedFeature]]:
        features = defaultdict(set)
        for feature in self.aggregated_features:
            features[feature.aggregate_over].add(feature)
        return features

    def with_sufix(self, sufix: str) -> "RetrievalRequest":
        return RetrievalRequest(
            name=f"{self.name}{sufix}",
            location=self.location,
            entities=self.entities,
            features=self.features,
            derived_features=self.derived_features,
            aggregated_features=self.aggregated_features,
            event_timestamp_request=self.event_timestamp_request,
        )

    def without_derived_features(self) -> "RetrievalRequest":
        assert not self.aggregated_features
        return RetrievalRequest(
            name=self.name,
            location=self.location,
            entities=self.entities,
            features=self.features,
            derived_features=set(),
            aggregated_features=set(),
            event_timestamp_request=self.event_timestamp_request,
        )

    def without_event_timestamp(
        self, name_sufix: str | None = None
    ) -> "RetrievalRequest":
        request = None
        if self.event_timestamp_request:
            request = EventTimestampRequest(
                self.event_timestamp_request.event_timestamp, None
            )

        name = self.name
        if name_sufix:
            name = f"{name}{name_sufix}"

        return RetrievalRequest(
            name=name,
            location=self.location,
            entities=self.entities,
            features=self.features,
            derived_features=self.derived_features,
            aggregated_features=self.aggregated_features,
            event_timestamp_request=request,
        )

    def with_event_timestamp_column(self, column: str) -> "RetrievalRequest":
        et_request = None
        if self.event_timestamp_request:
            et_request = EventTimestampRequest(
                self.event_timestamp_request.event_timestamp, column
            )
        return RetrievalRequest(
            name=self.name,
            location=self.location,
            entities=self.entities,
            features=self.features,
            derived_features=self.derived_features,
            aggregated_features=self.aggregated_features,
            event_timestamp_request=et_request,
        )

    @staticmethod
    def all_data() -> "RetrievalRequest":
        """
        This is a hack to tell aligned that we want all the data, and no filtering should be done.
        """
        return RetrievalRequest(
            name="",
            location=FeatureLocation.feature_view(""),
            entities=set(),
            features=set(),
            derived_features=set(),
            aggregated_features=set(),
            event_timestamp_request=None,
        )

    @staticmethod
    def combine(requests: list["RetrievalRequest"]) -> list["RetrievalRequest"]:
        grouped_requests: dict[FeatureLocation, RetrievalRequest] = {}
        returned_features: dict[FeatureLocation, set[Feature]] = {}
        entities = set()
        for request in requests:
            entities.update(request.entities)
            fv_name = request.location
            if fv_name not in grouped_requests:
                grouped_requests[fv_name] = RetrievalRequest(
                    name=request.name,
                    location=fv_name,
                    entities=request.entities,
                    features=request.features,
                    derived_features=request.derived_features,
                    aggregated_features=request.aggregated_features,
                    event_timestamp_request=request.event_timestamp_request,
                    features_to_include=request.features_to_include,
                )
                returned_features[fv_name] = request.returned_features
            else:
                grouped_requests[fv_name].derived_features.update(
                    request.derived_features
                )
                grouped_requests[fv_name].features.update(request.features)
                grouped_requests[fv_name].aggregated_features.update(
                    request.aggregated_features
                )
                grouped_requests[fv_name].entities.update(request.entities)
                returned_features[fv_name].update(request.returned_features)

        for request in grouped_requests.values():
            request.features_to_include = request.features_to_include.union(
                request.all_feature_names
                - {feature.name for feature in returned_features[request.location]}
            )

        return list(grouped_requests.values())

    def rename_entities(self, mapping: dict[str, str]) -> "RetrievalRequest":
        return RetrievalRequest(
            name=self.name,
            location=self.location,
            entities={
                entity.renamed(mapping.get(entity.name, entity.name))
                for entity in self.entities
            },
            features=self.features,
            derived_features=self.derived_features,
            aggregated_features=self.aggregated_features,
            event_timestamp_request=self.event_timestamp_request,
        )

    @staticmethod
    def unsafe_combine(requests: list["RetrievalRequest"]) -> "RetrievalRequest":
        entities = set()
        features = set()
        derived_features = set()
        aggregated_features = set()
        event_timestamp_request = None

        for request in requests:
            derived_features.update(request.derived_features)
            features.update(request.features)
            entities.update(request.entities)
            aggregated_features.update(request.aggregated_features)

            if event_timestamp_request is None:
                event_timestamp_request = request.event_timestamp_request

        return RetrievalRequest(
            name=requests[0].name,
            location=requests[0].location,
            entities=entities,
            features=features,
            derived_features=derived_features,
            aggregated_features=aggregated_features,
            event_timestamp_request=event_timestamp_request,
        )


@dataclass
class RequestResult(Codable):
    """
    Describes the returned response of a request
    """

    entities: set[Feature]
    features: set[Feature]
    event_timestamp: str | None

    @property
    def all_returned_columns(self) -> list[str]:
        columns = [entity.name for entity in self.entities]
        columns.extend([feat.name for feat in self.features])
        if self.event_timestamp:
            columns.append(self.event_timestamp)
        return columns

    @property
    def feature_columns(self) -> list[str]:
        return sorted(feature.name for feature in self.features)

    @property
    def entity_columns(self) -> list[str]:
        return [entity.name for entity in self.entities]

    def __add__(self, obj: "RequestResult") -> "RequestResult":
        return RequestResult(
            entities=self.entities.union(obj.entities),
            features=self.features.union(obj.features),
            event_timestamp=self.event_timestamp or obj.event_timestamp,
        )

    def filter_features(self, features_to_include: set[str]) -> "RequestResult":
        return RequestResult(
            entities=self.entities,
            features={
                feature
                for feature in self.features
                if feature.name in features_to_include
            },
            event_timestamp=self.event_timestamp,
        )

    @staticmethod
    def all_data() -> "RequestResult":
        return RequestResult(entities=set(), features=set(), event_timestamp=None)

    @staticmethod
    def from_request(request: RetrievalRequest) -> "RequestResult":
        return RequestResult(
            entities=request.entities,
            features=request.returned_features - request.entities,
            event_timestamp=request.event_timestamp.name
            if request.event_timestamp
            else None,
        )

    @staticmethod
    def from_request_list(requests: list[RetrievalRequest]) -> "RequestResult":
        request_len = len(requests)
        if request_len == 0:
            return RequestResult(entities=set(), features=set(), event_timestamp=None)
        elif request_len > 1:
            event_timestamp = None
            requests_with_event = [
                req.event_timestamp for req in requests if req.event_timestamp
            ]
            if requests_with_event:
                event_timestamp = requests_with_event[0].name
            return RequestResult(
                entities=set().union(*[request.entities for request in requests]),
                features=set().union(
                    *[
                        {
                            feature
                            for feature in request.returned_features
                            if feature.name in request.features_to_include
                        }
                        - request.entities
                        for request in requests
                    ]
                ),
                event_timestamp=event_timestamp,
            )
        else:
            return RequestResult.from_request(requests[0])

    @staticmethod
    def from_result_list(requests: list["RequestResult"]) -> "RequestResult":
        request_len = len(requests)
        if request_len == 0:
            return RequestResult(entities=set(), features=set(), event_timestamp=None)
        elif request_len > 1:
            event_timestamp = None
            requests_with_event = [
                req.event_timestamp for req in requests if req.event_timestamp
            ]
            if requests_with_event:
                event_timestamp = requests_with_event[0]
            return RequestResult(
                entities=set().union(*[request.entities for request in requests]),
                features=set().union(*[request.features for request in requests]),
                event_timestamp=event_timestamp,
            )
        else:
            return requests[0]

    def as_retrieval_request(
        self, name: str, location: FeatureLocation
    ) -> RetrievalRequest:
        return RetrievalRequest(
            name=name,
            location=location,
            entities=self.entities,
            features=self.features,
            derived_features=set(),
            aggregated_features=set(),
            event_timestamp_request=EventTimestampRequest(
                event_timestamp=EventTimestamp(name=self.event_timestamp),
                entity_column=None,
            )
            if self.event_timestamp
            else None,
        )


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

    location: FeatureLocation
    features_to_include: set[str]
    needed_requests: list[RetrievalRequest]

    @property
    def needs_event_timestamp(self) -> bool:
        return any(request.event_timestamp for request in self.needed_requests)

    @property
    def request_result(self) -> RequestResult:
        return RequestResult.from_request_list(self.needed_requests).filter_features(
            self.features_to_include
        )

    def entities(self) -> set[Feature]:
        features = set()
        for req in self.needed_requests:
            features.update(req.entities)
        return features

    def without_derived_features(self) -> "FeatureRequest":
        return FeatureRequest(
            location=self.location,
            features_to_include=self.features_to_include,
            needed_requests=[
                request.without_derived_features() for request in self.needed_requests
            ],
        )

    def without_event_timestamp(
        self, name_sufix: str | None = None
    ) -> "FeatureRequest":
        return FeatureRequest(
            location=self.location,
            features_to_include=self.features_to_include - {"event_timestamp"},
            needed_requests=[
                request.without_event_timestamp(name_sufix)
                for request in self.needed_requests
            ],
        )

    def with_sufix(self, sufix: str) -> "FeatureRequest":
        return FeatureRequest(
            location=self.location,
            features_to_include=self.features_to_include,
            needed_requests=[
                request.with_sufix(sufix) for request in self.needed_requests
            ],
        )

    def rename_entities(self, mappings: dict[str, str]) -> "FeatureRequest":
        return FeatureRequest(
            location=self.location,
            features_to_include=self.features_to_include,
            needed_requests=[
                request.rename_entities(mappings) for request in self.needed_requests
            ],
        )
