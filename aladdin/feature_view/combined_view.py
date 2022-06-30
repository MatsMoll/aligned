from abc import ABC, abstractproperty
from dataclasses import dataclass
from typing import Callable

from aladdin.codable import Codable
from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import Feature
from aladdin.feature_types import FeatureFactory, FeatureReferancable, TransformationFactory
from aladdin.feature_view.feature_view import CompiledFeatureView, FeatureSelectable, FeatureView, FVType
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest


@dataclass
class CombinedFeatureViewMetadata:
    name: str
    description: str | None = None
    tags: dict[str, str] | None = None
    owner: str | None = None


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


class CombinedFeatureView(ABC, FeatureSelectable):
    @abstractproperty
    def metadata(self) -> CombinedFeatureViewMetadata:
        pass

    @staticmethod
    def _needed_features(
        depending_on: list[FeatureReferancable], feature_views: dict[str, CompiledFeatureView]
    ) -> list[RetrivalRequest]:

        feature_refs: dict[CompiledFeatureView, set[str]] = {}

        for feature_dep in depending_on:
            view = feature_views[feature_dep.feature_view]
            feature_refs.setdefault(view, set()).add(feature_dep.name)

        return [
            feature_view.request_for(features).needed_requests[0]
            for feature_view, features in feature_refs.items()
        ]

    @classmethod
    def compile(cls) -> CompiledCombinedFeatureView:
        transformations: set[DerivedFeature] = set()
        metadata = cls().metadata
        var_names = [name for name in cls().__dir__() if not name.startswith('_')]

        requests: dict[str, list[RetrivalRequest]] = {}
        feature_view_deps: dict[str, CompiledFeatureView] = {}

        for var_name in var_names:
            feature = getattr(cls, var_name)
            if isinstance(feature, FeatureView):
                feature_view_deps[feature.metadata.name] = feature.compile()
            if isinstance(feature, TransformationFactory):
                feature._feature_view = metadata.name
                feature._name = var_name  # Needed in some cases for later inferance and features
                requests[var_name] = CombinedFeatureView._needed_features(
                    feature.using_features, feature_view_deps
                )

                sources = [
                    (feature_view_deps[request.feature_view_name], request) for request in requests[var_name]
                ]
                tran = feature.transformation(sources)
                derived = DerivedFeature(
                    name=var_name,
                    dtype=feature.feature._dtype,
                    depending_on={feature.feature_referance() for feature in feature.using_features},
                    transformation=tran,
                )
                transformations.add(derived)

        return CompiledCombinedFeatureView(
            name=metadata.name,
            features=transformations,
            feature_referances=requests,
        )

    @classmethod
    def select(
        cls: type[FVType], features: Callable[[type[FVType]], list[FeatureFactory]]
    ) -> RetrivalRequest:
        view: CompiledCombinedFeatureView = cls.compile()
        names = features(cls)
        return view.requests_for({feature.name for feature in names if feature.name})

    @classmethod
    def select_all(cls: type[FVType]) -> RetrivalRequest:
        return cls.compile().request_all
