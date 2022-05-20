from abc import ABC, abstractproperty
from collections import defaultdict
from dataclasses import dataclass
import re
from aladdin import FeatureView
from aladdin.derivied_feature import DerivedFeature
from aladdin.feature import Feature
from aladdin.feature_types import TransformationFactory
from aladdin.feature_view.feature_view import CompiledFeatureView
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.codable import Codable
from aladdin.feature_view.feature_view import FeatureSelectable, FVType
from aladdin.feature_types import FeatureFactory

from typing import Callable


@dataclass
class CombinedFeatureViewMetadata:
    name: str
    description: str | None = None
    tags: dict[str, str] | None = None
    owner: str | None = None


@dataclass
class CompiledCombinedFeatureView(Codable):
    name: str
    features: set[DerivedFeature] # FIXME: Should combine this and feature_referances into one class.
    feature_referances: dict[str, list[RetrivalRequest]]
    
    @property
    def request_all(self) -> RetrivalRequest:
        return list(self.feature_referances.values())[0]

    def requests_for(self, feature_names: set[str]) -> list[RetrivalRequest]:
        dependent_views: dict[str, RetrivalRequest] = {}
        all_entities = set()
        for feature in feature_names:
            if feature not in self.feature_referances.keys():
                raise ValueError(f"Invalid feature {feature} in {self.name}")
            requests = self.feature_referances[feature]
            for request in requests:
                if request.feature_view_name not in dependent_views:
                    dependent_views[request.feature_view_name] = RetrivalRequest(
                        feature_view_name=request.feature_view_name,
                        entities=request.entities,
                        features=set(),
                        derived_features=set()
                    )
                    all_entities.update(request.entities)
                current = dependent_views[request.feature_view_name]
                current.derived_features = current.derived_features.union(request.derived_features)
                current.features = current.features.union(request.features)
                dependent_views[request.feature_view_name] = current

        dependent_views[self.name] = RetrivalRequest( # Add the request we want
            feature_view_name=self.name,
            entities=all_entities,
            features=[],
            derived_features=[feature for feature in self.features if feature.name in feature_names]
        )

        return list(dependent_views.values())

    def __hash__(self) -> int:
        return hash(self.name)

class CombinedFeatureView(ABC, FeatureSelectable):
    
    @abstractproperty
    def metadata(self) -> CombinedFeatureViewMetadata:
        pass


    @staticmethod
    def _needed_features(feature: DerivedFeature, feature_views: dict[str, CompiledFeatureView]) -> list[RetrivalRequest]:

        feature_refs: dict[CompiledFeatureView, set[str]] = {}

        for feature_dep in feature.depending_on:
            view = feature_views[feature_dep.feature_view]
            feature_refs.setdefault(view, set()).add(feature_dep.name)

        return [
            feature_view.request_for(features) 
            for feature_view, features
            in feature_refs.items()
        ]
    
    @classmethod
    def compile(cls) -> CompiledCombinedFeatureView:
        transformations: set[DerivedFeature] = set()
        metadata = cls().metadata
        var_names = [name for name in cls().__dir__() if not name.startswith("_")]

        requests: dict[Feature, list[RetrivalRequest]] = {}
        feature_view_deps: dict[str, CompiledFeatureView] = {}
        
        for var_name in var_names:
            feature = getattr(cls, var_name)
            if isinstance(feature, FeatureView):
                feature_view_deps[feature.metadata.name] = feature.compile()
            if isinstance(feature, TransformationFactory):
                feature.feature_view = metadata.name
                feature.name = var_name # Needed in some cases for later inferance and features
                derived = DerivedFeature(
                    name=var_name,
                    dtype=feature.feature._dtype,
                    depending_on=[feature.feature_referance() for feature in feature.using_features],
                    transformation=feature.transformation
                )
                transformations.add(derived)
                requests[derived.name] = CombinedFeatureView._needed_features(derived, feature_view_deps)

        print(requests)

        return CompiledCombinedFeatureView(
            name=metadata.name,
            features=transformations,
            feature_referances=requests,
        )

    @classmethod
    def select(cls: type[FVType], features: Callable[[type[FVType]], list[FeatureFactory]]) -> RetrivalRequest:
        view = cls.compile()
        names = features(cls)
        return view.request_for({feature.name for feature in names if feature.name})

    @classmethod
    def select_all(cls: type[FVType]) -> RetrivalRequest:
        return cls.compile().request_all