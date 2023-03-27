import logging
from abc import ABC, abstractproperty
from dataclasses import dataclass

from aligned.compiler.feature_factory import FeatureFactory
from aligned.feature_view.feature_view import FeatureView
from aligned.request.retrival_request import RetrivalRequest
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import FeatureLocation
from aligned.schemas.feature_view import CompiledCombinedFeatureView, CompiledFeatureView

logger = logging.getLogger(__name__)


@dataclass
class CombinedFeatureViewMetadata:
    name: str
    description: str | None = None
    tags: dict[str, str] | None = None
    owner: str | None = None


class CombinedFeatureView(ABC):
    @abstractproperty
    def metadata(self) -> CombinedFeatureViewMetadata:
        pass

    @staticmethod
    def _needed_features(
        depending_on: list[FeatureFactory], feature_views: dict[FeatureLocation, CompiledFeatureView]
    ) -> list[RetrivalRequest]:

        feature_refs: dict[CompiledFeatureView, set[str]] = {}

        for feature_dep in depending_on:
            view = feature_views[feature_dep._location]
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
        feature_view_deps: dict[FeatureLocation, CompiledFeatureView] = {}

        for var_name in var_names:
            feature = getattr(cls, var_name)
            if isinstance(feature, FeatureView):
                # Needs to compile the view one more time. unfortunally..
                # not optimal as the view will be duplicated in the definition file
                feature_view_deps[FeatureLocation.feature_view(feature.metadata.name)] = feature.compile()
            if isinstance(feature, FeatureFactory):
                feature._location = FeatureLocation.combined_view(var_name)
                if not feature.transformation:
                    logger.info('Feature had no transformation, which do not make sense in a CombinedView')
                    continue
                requests[var_name] = CombinedFeatureView._needed_features(
                    feature.transformation.using_features, feature_view_deps
                )

                transformations.add(feature.compile())

        return CompiledCombinedFeatureView(
            name=metadata.name,
            features=transformations,
            feature_referances=requests,
        )
