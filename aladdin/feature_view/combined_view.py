import logging
from abc import ABC, abstractproperty
from dataclasses import dataclass
from typing import Callable

from aladdin.compiler.feature_factory import FeatureFactory
from aladdin.feature_view.feature_view import CompiledFeatureView, FeatureSelectable, FeatureView, FVType
from aladdin.request.retrival_request import FeatureRequest, RetrivalRequest
from aladdin.schemas.codable import Codable
from aladdin.schemas.derivied_feature import DerivedFeature
from aladdin.schemas.feature import Feature

logger = logging.getLogger(__name__)


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
        depending_on: list[FeatureFactory], feature_views: dict[str, CompiledFeatureView]
    ) -> list[RetrivalRequest]:

        feature_refs: dict[CompiledFeatureView, set[str]] = {}

        for feature_dep in depending_on:
            view = feature_views[feature_dep._feature_view]
            feature_refs.setdefault(view, set()).add(feature_dep.name)

        return [
            feature_view.request_for(features).needed_requests[0]
            for feature_view, features in feature_refs.items()
        ]

    @classmethod
    async def compile(cls) -> CompiledCombinedFeatureView:
        transformations: set[DerivedFeature] = set()
        metadata = cls().metadata
        var_names = [name for name in cls().__dir__() if not name.startswith('_')]

        requests: dict[str, list[RetrivalRequest]] = {}
        feature_view_deps: dict[str, CompiledFeatureView] = {}

        for var_name in var_names:
            feature = getattr(cls, var_name)
            if isinstance(feature, FeatureView):
                # Needs to compile the view one more time. unfortunally..
                # not optimal as the view will be duplicated in the definition file
                feature_view_deps[feature.metadata.name] = await feature.compile()
            if isinstance(feature, FeatureFactory):
                feature._feature_view = metadata.name
                feature._name = var_name  # Needed in some cases for later inferance and features
                if not feature.transformation:
                    logger.info('Feature had no transformation, which do not make sense in a CombinedView')
                    continue
                requests[var_name] = CombinedFeatureView._needed_features(
                    feature.transformation.using_features, feature_view_deps
                )

                transformations.add(await feature.compile(list(feature_view_deps.values())))

        return CompiledCombinedFeatureView(
            name=metadata.name,
            features=transformations,
            feature_referances=requests,
        )

    @classmethod
    async def compile_only_graph(cls) -> CompiledCombinedFeatureView:
        transformations: set[DerivedFeature] = set()
        metadata = cls().metadata
        var_names = [name for name in cls().__dir__() if not name.startswith('_')]

        requests: dict[str, list[RetrivalRequest]] = {}
        feature_view_deps: dict[str, CompiledFeatureView] = {}

        for var_name in var_names:
            feature = getattr(cls, var_name)
            if isinstance(feature, FeatureView):
                # Needs to compile the view one more time. unfortunally..
                # not optimal as the view will be duplicated in the definition file
                feature_view_deps[feature.metadata.name] = feature.compile_graph_only()
            if isinstance(feature, FeatureFactory):
                feature._feature_view = metadata.name
                feature._name = var_name  # Needed in some cases for later inferance and features
                if not feature.transformation:
                    logger.info('Feature had no transformation, which do not make sense in a CombinedView')
                    continue
                requests[var_name] = CombinedFeatureView._needed_features(
                    feature.transformation.using_features, feature_view_deps
                )

                transformations.add(feature.compile_graph_only())

        return CompiledCombinedFeatureView(
            name=metadata.name,
            features=transformations,
            feature_referances=requests,
        )

    @classmethod
    def select(
        cls: type[FVType], features: Callable[[type[FVType]], list[FeatureFactory]]
    ) -> RetrivalRequest:
        view: CompiledCombinedFeatureView = cls.compile_only_graph()
        names = features(cls)
        return view.requests_for({feat.name for feat in names})

    @classmethod
    def select_all(cls: type[FVType]) -> RetrivalRequest:
        return cls.compile().request_all
