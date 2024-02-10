import logging
from abc import ABC, abstractproperty
from dataclasses import dataclass
from typing import Generic, TypeVar, Any, Type, Callable, TYPE_CHECKING

from aligned.compiler.feature_factory import FeatureFactory
from aligned.feature_view.feature_view import FeatureView
from aligned.request.retrival_request import RetrivalRequest
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import FeatureLocation
from aligned.schemas.feature_view import CompiledCombinedFeatureView, CompiledFeatureView

if TYPE_CHECKING:
    from aligned.feature_store import FeatureViewStore

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CombinedFeatureViewMetadata:
    name: str
    description: str | None = None
    tags: dict[str, str] | None = None
    owner: str | None = None


@dataclass
class CombinedFeatureViewWrapper(Generic[T]):

    metadata: CombinedFeatureViewMetadata
    view: Type[T]

    def __call__(self) -> T:
        # Needed to make sure that the `location` is set in the view's features
        _ = self.compile()
        return self.view()

    def compile(self) -> CompiledCombinedFeatureView:
        return CombinedFeatureView.compile_with_metadata(self.view(), self.metadata)

    def query(self) -> 'FeatureViewStore':
        """Makes it possible to query the feature view for features

        ```python
        @feature_view(...)
        class SomeView:

            id = Int32().as_entity()

            a = Int32()
            b = Int32()

        data = await SomeView.query().features_for({
            "id": [1, 2, 3],
        }).to_pandas()
        ```

        Returns:
            FeatureViewStore: Returns a queryable `FeatureViewStore` containing the feature view
        """
        from aligned import FeatureStore

        store = FeatureStore.experimental()
        store.add_combined_view(self.compile())
        return store.feature_view(self.metadata.name)

    async def process(self, data: dict[str, list[Any]]) -> list[dict]:
        df = await self.query().process_input(data).to_lazy_polars()
        return df.collect().to_dicts()


def combined_feature_view(
    name: str, description: str, tags: dict[str, str] | None = None, owner: str | None = None
) -> Callable[[Type[T]], CombinedFeatureViewWrapper[T]]:
    """
    Wraps a view as a combined view

    ```python
    @combined_feature_view(
        name="my_combined_view",
        description="some description"
    )
    class MyView:

        other = OtherView()
        another = AnotherView()

        y = other.x * another.y
    ```
    """

    def decorator(cls: Type[T]) -> CombinedFeatureViewWrapper[T]:
        return CombinedFeatureViewWrapper(
            CombinedFeatureViewMetadata(name, description, tags=tags, owner=owner), cls
        )

    return decorator


class CombinedFeatureView(ABC):
    @abstractproperty
    def metadata(self) -> CombinedFeatureViewMetadata:
        raise NotImplementedError(f'Need to add a metadata field to in {self}')

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
        instance = cls()
        return CombinedFeatureView.compile_with_metadata(instance, instance.metadata)

    @staticmethod
    def compile_with_metadata(
        view: Any, metadata: CombinedFeatureViewMetadata
    ) -> CompiledCombinedFeatureView:
        transformations: set[DerivedFeature] = set()
        var_names = [name for name in view.__dir__() if not name.startswith('_')]

        requests: dict[str, list[RetrivalRequest]] = {}
        feature_view_deps: dict[FeatureLocation, CompiledFeatureView] = {}

        for var_name in var_names:
            feature = getattr(view, var_name)
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
