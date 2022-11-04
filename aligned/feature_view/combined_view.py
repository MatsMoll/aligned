import logging
from abc import ABC, abstractproperty
from dataclasses import dataclass
from typing import Callable

from aligned.compiler.feature_factory import FeatureFactory
from aligned.feature_view.feature_view import FeatureSelectable, FeatureView, FVType
from aligned.request.retrival_request import RetrivalRequest
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature_view import CompiledCombinedFeatureView, CompiledFeatureView

logger = logging.getLogger(__name__)


@dataclass
class CombinedFeatureViewMetadata:
    name: str
    description: str | None = None
    tags: dict[str, str] | None = None
    owner: str | None = None


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
    def compile_only_graph(cls) -> CompiledCombinedFeatureView:
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
        view: CompiledCombinedFeatureView = cls.compile_only_graph()  # type: ignore[attr-defined]
        names = features(cls)
        return view.requests_for({feat.name for feat in names})

    @classmethod
    def select_all(cls: type[FVType]) -> RetrivalRequest:
        return cls.compile_only_graph().request_all  # type: ignore[attr-defined]
