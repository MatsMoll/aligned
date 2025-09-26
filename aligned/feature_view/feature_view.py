from __future__ import annotations

import copy
import logging
import polars as pl

from datetime import timedelta
from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Literal,
    Sequence,
    TypeVar,
    Generic,
    Type,
    Callable,
    Union,
    overload,
)

from aligned.lazy_imports import pandas as pd
from aligned.compiler.feature_factory import (
    AggregationTransformationFactory,
    Embedding,
    EventTimestamp,
    Bool,
    FeatureReferencable,
)
from aligned.data_source.batch_data_source import (
    AsBatchSource,
    CodableBatchDataSource,
    CustomMethodDataSource,
    JoinAsofDataSource,
    JoinDataSource,
    join_asof_source,
    join_source,
    resolve_keys,
)
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.local.job import LiteralRetrievalJob
from aligned.request.retrieval_request import RetrievalRequest
from aligned.retrieval_job import (
    ConvertableToRetrievalJob,
    FilterRepresentable,
    RetrievalJob,
)
from aligned.schemas.derivied_feature import (
    AggregatedFeature,
)
from aligned.schemas.feature import FeatureLocation, FeatureReference, StaticFeatureTags
from aligned.schemas.feature_view import CompiledFeatureView, Contact
from aligned.compiler.feature_factory import FeatureFactory
from datetime import datetime

from aligned.schemas.transformation import Expression

if TYPE_CHECKING:
    from aligned.feature_store import FeatureViewStore
    from aligned.validation.interface import Validator

# Enables code completion in the select method
T = TypeVar("T")

ConvertableData = TypeVar("ConvertableData", dict, pl.DataFrame, "pd.DataFrame")


logger = logging.getLogger(__name__)


@dataclass
class FeatureViewMetadata:
    name: str
    source: CodableBatchDataSource
    description: str | None = field(default=None)
    stream_source: StreamDataSource | None = field(default=None)
    application_source: CodableBatchDataSource | None = field(default=None)
    materialized_source: CodableBatchDataSource | None = field(default=None)
    materialize_from: datetime | None = field(default=None)
    contacts: list[Contact] | None = field(default=None)
    tags: list[str] | None = field(default=None)
    acceptable_freshness: timedelta | None = field(default=None)
    unacceptable_freshness: timedelta | None = field(default=None)

    @staticmethod
    def from_compiled(view: CompiledFeatureView) -> FeatureViewMetadata:
        return FeatureViewMetadata(
            name=view.name,
            description=view.description,
            tags=view.tags,
            source=view.source,
            stream_source=view.stream_data_source,
            application_source=view.application_source,
            materialized_source=view.materialized_source,
            materialize_from=view.materialize_from,
            acceptable_freshness=view.acceptable_freshness,
            unacceptable_freshness=view.unacceptable_freshness,
        )


PureLoadFunctions = Union[
    Callable[[RetrievalRequest, datetime, datetime], Awaitable[pl.LazyFrame]],
    Callable[[RetrievalRequest, int | None], Awaitable[pl.LazyFrame]],
    Callable[[RetrievalRequest, int], Awaitable[pl.LazyFrame]],
]


def resolve_source(
    source: CodableBatchDataSource
    | AsBatchSource
    | FeatureViewWrapper
    | PureLoadFunctions
    | None,
) -> CodableBatchDataSource:
    if source is None:
        from aligned.sources.in_mem_source import InMemorySource

        return InMemorySource.from_values({})

    if isinstance(source, AsBatchSource):
        return source.as_source()

    if isinstance(source, FeatureViewWrapper):
        from aligned.schemas.feature_view import FeatureViewReferenceSource

        compiled = source.compile()
        return FeatureViewReferenceSource(
            compiled, FeatureLocation.feature_view(compiled.name)
        )

    elif isinstance(source, CodableBatchDataSource):
        return source

    import inspect

    signature = inspect.signature(source)
    params = signature.parameters

    if len(params) == 1:
        return CustomMethodDataSource.from_load(source)  # type: ignore

    if len(params) == 3 and "request" in params and "start_date" in params:
        return CustomMethodDataSource.from_methods(all_between_dates=source)  # type: ignore

    if len(params) == 2 and "request" in params and "limit" in params:
        return CustomMethodDataSource.from_methods(all_data=source)  # type: ignore

    raise ValueError(f"Unable to use function with signature {signature} as source.")


def set_location_for_features_in(view: Any, location: FeatureLocation) -> Any:
    for attribute in dir(view):
        if attribute.startswith("__"):
            continue

        value = getattr(view, attribute)
        if isinstance(value, FeatureFactory):
            value._location = location
            copied = copy.copy(value)

            setattr(view, attribute, copied)
    return view


@dataclass
class FeatureViewWrapper(Generic[T]):
    metadata: FeatureViewMetadata
    view: T

    @property
    def location(self) -> FeatureLocation:
        return FeatureLocation.feature_view(self.metadata.name)

    @property
    def request(self) -> RetrievalRequest:
        return self.compile().retrieval_request

    def __call__(self) -> T:
        view = copy.deepcopy(self.view)
        view = set_location_for_features_in(
            view, FeatureLocation.feature_view(self.metadata.name)
        )
        _ = FeatureView.compile_with_metadata(view, self.metadata)
        setattr(view, "__view_wrapper__", self)
        return view

    def compile(self) -> CompiledFeatureView:
        view = copy.deepcopy(self.view)
        view = set_location_for_features_in(
            view, FeatureLocation.feature_view(self.metadata.name)
        )
        return FeatureView.compile_with_metadata(view, self.metadata)

    def empty(self) -> RetrievalJob:
        view = self.compile()
        req = view.retrieval_request
        return LiteralRetrievalJob(
            pl.DataFrame(
                data=[],
                schema={
                    feature.name: feature.dtype.polars_type
                    for feature in req.all_returned_features
                },
            ),
            requests=[req],
        )

    def vstack(
        self,
        source: CodableBatchDataSource | FeatureViewWrapper,
        source_column: str | None = None,
    ) -> CodableBatchDataSource:
        from aligned.data_source.batch_data_source import StackSource

        return StackSource(
            top=resolve_source(self),
            bottom=resolve_source(source),
            source_column=source_column,
        )

    def feature_references(
        self,
        exclude: Sequence[str]
        | Callable[[T], Sequence[FeatureReferencable] | FeatureReferencable]
        | None = None,
        include: Sequence[str]
        | Callable[[T], Sequence[FeatureReferencable] | FeatureReferencable]
        | None = None,
    ) -> list[FeatureReference]:
        req = self.request

        if exclude is not None:
            if callable(exclude):
                refs = exclude(self.view)
                if isinstance(refs, FeatureReferencable):
                    refs = [refs]
                feature_names = [feat.feature_reference().name for feat in refs]
            else:
                feature_names = list(exclude)

            return [
                feat.as_reference(req.location)
                for feat in req.all_features
                if feat.name not in feature_names
            ]

        if include is not None:
            if callable(include):
                refs = include(self.view)
                if isinstance(refs, FeatureReferencable):
                    refs = [refs]
                feature_names = [feat.feature_reference().name for feat in refs]
            else:
                feature_names = list(include)

            return [
                feat.as_reference(req.location)
                for feat in req.all_features
                if feat.name in feature_names
            ]

        return [feat.as_reference(req.location) for feat in req.all_features]

    def filter(
        self,
        name: str,
        where: Callable[[T], Bool] | FilterRepresentable,
        materialize_source: CodableBatchDataSource | None = None,
    ) -> FeatureViewWrapper[T]:
        from aligned.data_source.batch_data_source import FilteredDataSource
        from aligned.schemas.feature_view import FeatureViewReferenceSource

        meta = copy.deepcopy(self.metadata)
        meta.name = name
        meta.materialized_source = materialize_source

        main_source = FeatureViewReferenceSource(
            self.compile(), FeatureLocation.feature_view(self.metadata.name)
        )

        if callable(where):
            condition = where(self.__call__())
        else:
            condition = where

        exp = Expression.from_value(condition)

        meta.source = FilteredDataSource(main_source, exp)
        return FeatureViewWrapper(metadata=meta, view=self.view)

    def join(
        self,
        view: Any,
        on: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        on_left: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        on_right: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        how: Literal["inner", "left", "outer"] = "inner",
    ) -> JoinDataSource:
        from aligned.schemas.feature_view import FeatureViewReferenceSource

        compiled_view = self.compile()
        source = FeatureViewReferenceSource(
            compiled_view, FeatureLocation.feature_view(compiled_view.name)
        )

        if on:
            on_left = on
            on_right = on

        return join_source(
            source,
            view,
            on_left,
            on_right,
            how,
            left_request=compiled_view.request_all.needed_requests[0],
        )

    def join_asof(
        self, view: Any, on: str | FeatureFactory | list[str] | list[FeatureFactory]
    ) -> JoinAsofDataSource:
        from aligned.schemas.feature_view import FeatureViewReferenceSource

        compiled_view = self.compile()
        source = FeatureViewReferenceSource(
            compiled_view, FeatureLocation.feature_view(compiled_view.name)
        )

        left_on = None
        right_on = None
        if on:
            left_on = resolve_keys(on)
            right_on = left_on

        return join_asof_source(
            source,
            left_request=compiled_view.request_all.needed_requests[0],
            view=view,
            left_on=left_on,
            right_on=right_on,
        )

    def with_schema(
        self,
        name: str,
        source: CodableBatchDataSource | FeatureViewWrapper,
        materialized_source: CodableBatchDataSource | None = None,
        entities: dict[str, FeatureFactory] | None = None,
        additional_features: dict[str, FeatureFactory] | None = None,
        copy_default_values: bool = False,
        copy_transformations: bool = False,
    ) -> FeatureViewWrapper[T]:
        meta = copy.deepcopy(self.metadata)
        meta.name = name
        meta.source = resolve_source(source)
        meta.materialized_source = None

        if materialized_source:
            meta.materialized_source = resolve_source(materialized_source)

        view = copy.deepcopy(self.view)
        compiled = self.compile()

        for agg_feature in compiled.aggregated_features:
            if agg_feature.name.isdigit():
                continue
            org_feature: FeatureFactory = getattr(
                view, agg_feature.derived_feature.name
            )
            feature = org_feature.copy_type()
            feature.transformation = None
            feature.tags = set(agg_feature.derived_feature.tags or [])
            if copy_transformations:
                feature.transformation = copy.deepcopy(org_feature.transformation)
            if copy_default_values:
                feature._default_value = org_feature._default_value
            setattr(view, agg_feature.derived_feature.name, feature)

        for derived_feature in compiled.derived_features:
            if derived_feature.name.isdigit():
                continue
            org_feature: FeatureFactory = getattr(view, derived_feature.name)
            feature = org_feature.copy_type()
            feature.transformation = None
            if copy_transformations:
                feature.transformation = copy.deepcopy(org_feature.transformation)
            feature.tags = set(derived_feature.tags or [])
            if copy_default_values:
                feature._default_value = org_feature._default_value
            setattr(view, derived_feature.name, feature)

        if entities is not None:
            for name, feature in entities.items():
                setattr(view, name, feature.as_entity())  # type: ignore

            for entity in compiled.entities:
                if entity.name in entities:
                    continue

                setattr(view, entity.name, None)

        if additional_features is not None:
            for name, feature in additional_features.items():
                setattr(view, name, feature)

        return FeatureViewWrapper(meta, view)

    def with_source(
        self,
        named: str,
        source: CodableBatchDataSource | FeatureViewWrapper,
        materialized_source: CodableBatchDataSource | None = None,
    ) -> FeatureViewWrapper[T]:
        meta = copy.deepcopy(self.metadata)
        meta.name = named
        meta.source = resolve_source(source)
        meta.materialized_source = None

        if materialized_source:
            meta.materialized_source = resolve_source(materialized_source)

        return FeatureViewWrapper(metadata=meta, view=self.view)

    def with_entity_renaming(
        self, named: str, renames: dict[str, str] | str
    ) -> FeatureViewWrapper[T]:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable

        compiled_view = self.compile()

        meta = copy.deepcopy(self.metadata)
        meta.name = named

        all_data_sources = [
            meta.source,
            meta.materialized_source,
            meta.application_source,
            meta.stream_source,
        ]

        if isinstance(renames, str):
            if not len(compiled_view.entities) == 1:
                raise ValueError(
                    f"Renaming entities for {compiled_view.name} with a string '{renames}'"
                    "is impossible. Need to setup a dict to know which entity to rename."
                )

            entity_name = list(compiled_view.entitiy_names)[0]
            renames = {entity_name: renames}

        for source in all_data_sources:
            if not isinstance(source, ColumnFeatureMappable):
                logger.info(
                    f"Source {type(source)} do not conform to ColumnFeatureMappable,"
                    "which could lead to problems"
                )
                continue
            for key, value in renames.items():
                existing_map = source.mapping_keys.get(key)

                if existing_map:
                    source.mapping_keys[value] = existing_map
                else:
                    source.mapping_keys[value] = key

        return FeatureViewWrapper(metadata=meta, view=self.view)

    def query(self) -> FeatureViewStore:
        """Makes it possible to query the feature view for features

        ```python
        class SomeView(FeatureView):

            metadata = ...

            id = Int32().as_entity()

            a = Int32()
            b = Int32()

        data = await SomeView.query().features_for({
            "id": [1, 2, 3]
        }).to_pandas()
        ```

        Returns:
            FeatureViewStore: Returns a queryable `FeatureViewStore` containing the feature view
        """
        from aligned import ContractStore

        store = ContractStore.experimental()
        store.add_compiled_view(self.compile())
        return store.feature_view(self.metadata.name)

    async def materialize_all(self) -> None:
        """
        Loads all the data from source, and writes it to the materialized source
        """
        if not self.metadata.materialized_source:
            return

        store = self.query()
        await store.overwrite(store.using_source(self.metadata.source).all())

    def process_input(self, data: ConvertableToRetrievalJob) -> RetrievalJob:
        return self.query().process_input(data)

    async def process(self, data: ConvertableToRetrievalJob) -> list[dict]:
        df = await self.query().process_input(data).to_lazy_polars()
        return df.collect().to_dicts()

    async def freshness_in_source(
        self, source: CodableBatchDataSource
    ) -> datetime | None:
        """
        Returns the freshest datetime for a provided source

        ```python
        psql_source = PostgreSQLConfig.localhost()
        my_data = FileSource.parquet_at("data.parquet")

        @feature_view(
            name="my_view",
            batch_source=psql_source.table("my_view")
        )
        class MyView:

            id = Int32().as_entity()

            updated_at = EventTimestamp()

            x = Float()


        await MyView.freshness_in_source(my_data)

        >>> datetime.datetime(2016, 1, 11, 23, 32, 55)
        ```

        Args:
            source (BatchDataSource): The source to get the freshness for
        """
        compiled = self.compile()
        return await FeatureView.freshness_in_source(compiled, source)

    async def freshness_in_batch_source(self) -> datetime | None:
        """
        Returns the freshest datetime for a provided source

        ```python
        psql_source = PostgreSQLConfig.localhost()

        @feature_view(
            name="my_view",
            batch_source=psql_source.table("my_view")
        )
        class MyView:

            id = Int32().as_entity()

            updated_at = EventTimestamp()

            x = Float()


        await MyView.freshness_in_batch_source()

        >>> datetime.datetime(2016, 1, 11, 23, 32, 55)
        ```
        """
        compiled = self.compile()
        return await FeatureView.freshness_in_source(compiled, compiled.source)

    def from_data(self, data: ConvertableToRetrievalJob) -> RetrievalJob:
        request = self.compile().request_all
        return RetrievalJob.from_convertable(data, request)

    def examples(self, data: ConvertableToRetrievalJob) -> RetrievalJob:
        from aligned.sources.random_source import RandomDataSource

        request = self.compile().request_all
        return RandomDataSource().features_for(
            RetrievalJob.from_convertable(data, request), request.needed_requests[0]
        )

    def n_examples(self, n: int) -> RetrievalJob:
        from aligned.sources.random_source import RandomDataSource

        request = self.compile().request_all
        return RandomDataSource(default_data_size=n).all(request.request_result, n)

    @overload
    def drop_invalid(
        self, values: pl.LazyFrame, validator: Validator | None = None
    ) -> pl.LazyFrame: ...

    @overload
    def drop_invalid(
        self, values: pl.DataFrame, validator: Validator | None = None
    ) -> pl.DataFrame: ...

    @overload
    def drop_invalid(
        self, values: pd.DataFrame, validator: Validator | None = None
    ) -> pd.DataFrame: ...

    @overload
    def drop_invalid(
        self, values: dict[str, list], validator: Validator | None = None
    ) -> dict[str, list]: ...

    @overload
    def drop_invalid(
        self, values: list[dict[str, Any]], validator: Validator | None = None
    ) -> list[dict[str, Any]]: ...

    def drop_invalid(
        self, values: ConvertableToRetrievalJob, validator: Validator | None = None
    ) -> ConvertableToRetrievalJob:
        from aligned.retrieval_job import DropInvalidJob

        if validator is None:
            from aligned.validation.interface import PolarsValidator

            validator = PolarsValidator()

        features = list(
            DropInvalidJob.features_to_validate(
                self.compile().request_all.needed_requests
            )
        )

        if isinstance(values, pl.LazyFrame):
            return validator.validate_polars(features, values)
        elif isinstance(values, pl.DataFrame):
            df = values.lazy()
            return validator.validate_polars(features, df).collect()
        elif isinstance(values, dict):
            df = pl.DataFrame(values).lazy()
            return (
                validator.validate_polars(features, df)
                .collect()
                .to_dict(as_series=False)
            )
        elif isinstance(values, list):
            df = pl.DataFrame(values).lazy()
            return validator.validate_polars(features, df).collect().to_dicts()
        elif isinstance(values, pd.DataFrame):
            df = pl.from_pandas(values).lazy()
            return validator.validate_polars(features, df).collect().to_pandas()
        else:
            raise ValueError(f"Unable to convert {type(values)}")

    def as_source(
        self, renames: dict[str, str] | None = None
    ) -> CodableBatchDataSource:
        from aligned.schemas.feature_view import FeatureViewReferenceSource

        return FeatureViewReferenceSource(
            self.compile(),
            FeatureLocation.feature_view(self.metadata.name),
            renames=renames or {},
        )


BatchSourceable = (
    CodableBatchDataSource | FeatureViewWrapper | PureLoadFunctions | AsBatchSource
)


def feature_view(
    source: BatchSourceable | None = None,
    name: str | None = None,
    description: str | None = None,
    stream_source: StreamDataSource | None = None,
    application_source: BatchSourceable | None = None,
    materialized_source: BatchSourceable | None = None,
    materialize_from: datetime | None = None,
    contacts: list[str] | list[Contact] | None = None,
    tags: list[str] | None = None,
    acceptable_freshness: timedelta | None = None,
    unacceptable_freshness: timedelta | None = None,
) -> Callable[[Type[T]], FeatureViewWrapper[T]]:
    if contacts is not None:
        contacts = [
            Contact(name=cont) if isinstance(cont, str) else cont for cont in contacts
        ]

    def decorator(cls: Type[T]) -> FeatureViewWrapper[T]:
        from aligned.sources.renamer import camel_to_snake_case

        used_name = name or camel_to_snake_case(str(cls.__name__))
        used_description = description or str(cls.__doc__)

        metadata = FeatureViewMetadata(
            used_name,
            resolve_source(source),
            description=used_description,
            stream_source=stream_source,
            application_source=None
            if application_source is None
            else resolve_source(application_source),
            materialized_source=None
            if materialized_source is None
            else resolve_source(materialized_source),
            materialize_from=materialize_from,
            contacts=contacts,
            tags=tags,
            acceptable_freshness=acceptable_freshness,
            unacceptable_freshness=unacceptable_freshness,
        )
        return FeatureViewWrapper(metadata, cls())

    return decorator


data_contract = feature_view


class FeatureView(ABC):
    """
    A collection of features, and a way to combine them.

    This should contain the core features, and might contain derived features (aka. transformations).
    """

    @abstractproperty
    def metadata(self) -> FeatureViewMetadata:
        raise NotImplementedError()

    @staticmethod
    def metadata_with(
        name: str,
        batch_source: CodableBatchDataSource,
        description: str | None = None,
        stream_source: StreamDataSource | None = None,
        application_source: CodableBatchDataSource | None = None,
        staging_source: CodableBatchDataSource | None = None,
        contacts: list[Contact] | None = None,
        tags: list[str] | None = None,
    ) -> FeatureViewMetadata:
        from aligned import HttpStreamSource

        return FeatureViewMetadata(
            name,
            batch_source,
            description,
            stream_source or HttpStreamSource(name),
            application_source=application_source,
            materialized_source=staging_source,
            contacts=contacts,
            tags=tags,
        )

    @classmethod
    def compile(cls) -> CompiledFeatureView:
        return cls().compile_instance()

    def compile_instance(self) -> CompiledFeatureView:
        return FeatureView.compile_with_metadata(self, self.metadata)

    @classmethod
    async def batch_source_freshness(cls) -> datetime | None:
        """
        Returns the freshest datetime for the batch data source
        """
        compiled = cls().compile_instance()
        return await FeatureView.freshness_in_source(compiled, compiled.source)

    @staticmethod
    async def freshness_in_source(
        view: CompiledFeatureView, source: CodableBatchDataSource
    ) -> datetime | None:
        if not view.event_timestamp:
            raise ValueError(
                f"The feature view: {view.name}, needs an event timestamp",
                "to compute the freshness of a source",
            )
        return await source.freshness(view.event_timestamp.as_feature())

    @staticmethod
    def compile_with_metadata(
        feature_view: Any, metadata: FeatureViewMetadata
    ) -> CompiledFeatureView:
        from aligned.compiler.feature_factory import FeatureFactory

        var_names = [
            name for name in feature_view.__dir__() if not name.startswith("_")
        ]

        view = CompiledFeatureView(
            name=metadata.name,
            description=metadata.description,
            tags=metadata.tags,
            contacts=metadata.contacts,
            source=metadata.source,
            entities=set(),
            features=set(),
            derived_features=set(),
            aggregated_features=set(),
            event_timestamp=None,
            stream_data_source=metadata.stream_source,
            application_source=metadata.application_source,
            materialized_source=metadata.materialized_source,
            materialize_from=metadata.materialize_from,
            acceptable_freshness=metadata.acceptable_freshness,
            unacceptable_freshness=metadata.unacceptable_freshness,
            indexes=[],
        )
        aggregations: list[FeatureFactory] = []

        for var_name in var_names:
            feature = getattr(feature_view, var_name)

            if not isinstance(feature, FeatureFactory):
                continue

            feature._name = var_name
            feature._location = FeatureLocation.feature_view(metadata.name)
            compiled_feature = feature.feature()

            if isinstance(feature, Embedding) and feature.indexes:
                view.indexes.extend(
                    [
                        index.compile(
                            feature._location, compiled_feature, view.entities
                        )
                        for index in feature.indexes
                    ]
                )

            if feature.tags and StaticFeatureTags.is_entity in feature.tags:
                view.entities.add(compiled_feature)

                if feature.transformation:
                    feature._name = var_name
                    feature._location = FeatureLocation.feature_view(metadata.name)
                else:
                    continue

            if feature.transformation:
                # Adding features that is not stored in the view
                # e.g:
                # class SomeView(FeatureView):
                #     ...
                #     x, y = Bool(), Bool()
                #     z = (x & y) | x
                #
                # Here will (x & y)'s result be a 'hidden' feature
                feature_deps = [
                    (feat.depth(), feat) for feat in feature.feature_dependencies()
                ]
                hidden_features: list[FeatureFactory] = []

                # Sorting by key so the "core" features are first
                # And then making it possible for other features to reference them
                def sort_key(x: tuple[int, FeatureFactory]) -> int:
                    return x[0]

                for depth, feature_dep in sorted(feature_deps, key=sort_key):
                    if not feature_dep._location:
                        feature_dep._location = FeatureLocation.feature_view(
                            metadata.name
                        )
                    elif feature_dep._location.name != metadata.name:
                        continue

                    if feature_dep._name:
                        feat_dep = feature_dep.feature()
                        if feat_dep in view.features or feat_dep in view.entities:
                            continue

                    if depth == 0:
                        if not feature_dep._name:
                            feature_dep._name = var_name

                        feat_dep = feature_dep.feature()
                        view.features.add(feat_dep)
                        continue

                    if not feature_dep._name:
                        hidden_features.append(feature_dep)
                        continue

                    if isinstance(
                        feature_dep.transformation, AggregationTransformationFactory
                    ):
                        aggregations.append(feature_dep)
                    else:
                        feature_graph = (
                            feature_dep.compile()
                        )  # Should decide on which payload to send
                        if feature_graph in view.derived_features:
                            continue

                        view.derived_features.add(feature_dep.compile())

                if isinstance(feature.transformation, AggregationTransformationFactory):
                    aggregations.append(feature)
                else:
                    transformations = []
                    deps: set[FeatureReference] = set()

                    if hidden_features:
                        for index, feat in enumerate(hidden_features):
                            sub_name = str(index)
                            feat._name = sub_name
                            comp = feat.compile()
                            transformations.append((comp.transformation, sub_name))
                            deps.update(comp.depending_on)

                    compiled_der = feature.compile()

                    if hidden_features:
                        for feat in hidden_features:
                            # Clean up the name so they are not detected as actual features
                            feat._name = None

                    if transformations:
                        from aligned.schemas.transformation import MultiTransformation

                        transformations.append((compiled_der.transformation, None))
                        compiled_der.transformation = MultiTransformation(
                            transformations
                        )
                        compiled_der.depending_on = {
                            dep
                            for dep in deps
                            # Aka only non hidden features
                            if not dep.name.isdigit()
                        }

                    view.derived_features.add(
                        compiled_der
                    )  # Should decide on which payload to send

            elif isinstance(feature, EventTimestamp):
                if view.event_timestamp is not None:
                    raise Exception(
                        "Can only have one EventTimestamp for each"
                        " FeatureViewDefinition. Check that this is the case for"
                        f" {type(view).__name__}"
                    )
                view.event_timestamp = feature.event_timestamp()
            else:
                view.features.add(compiled_feature)

        loc = FeatureLocation.feature_view(view.name)
        aggregation_group_by = [
            FeatureReference(entity.name, loc) for entity in view.entities
        ]
        event_timestamp_ref = (
            FeatureReference(view.event_timestamp.name, loc)
            if view.event_timestamp
            else None
        )

        for aggr in aggregations:
            agg_trans = aggr.transformation
            if not isinstance(agg_trans, AggregationTransformationFactory):
                continue

            config = agg_trans.aggregate_over(aggregation_group_by, event_timestamp_ref)
            feature = aggr.compile()
            feat = AggregatedFeature(
                derived_feature=feature,
                aggregate_over=config,
            )
            view.aggregated_features.add(feat)

        view.source = view.source.with_view(view)

        if view.materialized_source:
            view.materialized_source = view.materialized_source.with_view(view)

        return view

    @classmethod
    def query(cls) -> FeatureViewStore:
        """Makes it possible to query the feature view for features

        ```python
        class SomeView(FeatureView):

            metadata = ...

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
        from aligned import ContractStore

        self = cls()
        store = ContractStore.experimental()
        store.add_feature_view(self)
        return store.feature_view(self.metadata.name)

    @classmethod
    async def process(cls, data: dict[str, list[Any]]) -> list[dict]:
        df = await cls.query().process_input(data).to_lazy_polars()
        return df.collect().to_dicts()

    @staticmethod
    def feature_view_code_template(
        schema: dict[str, FeatureFactory],
        batch_source_code: str,
        view_name: str,
        imports: str | None = None,
    ) -> str:
        """Setup the code needed to represent the data source as a feature view

        ```python

        source = FileSource.parquet_at("file.parquet")
        schema = await source.schema()
        FeatureView.feature_view_code_template(schema, batch_source_code=f"{source}", view_name="my_view")

        >>> \"\"\"from aligned import feature_view, String, Int64, Float

        @feature_view(
            name="titanic",
            description="some description",
            source=FileSource.parquest("my_path.parquet")
            stream_source=None,
        )
        class MyView:

            Passenger_id = Int64()
            Survived = Int64()
            Pclass = Int64()
            Name = String()
            Sex = String()
            Age = Float()
            Sibsp = Int64()
            Parch = Int64()
            Ticket = String()
            Fare = Float()
            Cabin = String()
            Embarked = String()\"\"\"
        ```

        Returns:
            str: The code needed to setup a basic feature view
        """
        data_types: set[str] = set()
        feature_code = ""
        for name, dtype in schema.items():
            type_name = dtype.__class__.__name__
            data_types.add(type_name)
            feature_code += f"{name} = {type_name}()\n    "

        all_types = ", ".join(data_types)

        return f"""
from aligned import feature_view, {all_types}
{imports or ''}

@feature_view(
    name="{view_name}",
    description="some description",
    source={batch_source_code},
    stream_source=None,
)
class MyView:

    {feature_code}
    """


def check_schema() -> Callable:
    """
    A wrapper that checks the schema of data frames given a feature view or model contract.


    ```python
    @feature_view(...)
    class MyView:
        id = Int32().as_entity()
        name = String()

    @check_schema()
    def my_function(data: Annotated[pd.DataFrame, MyView]):
        ...


    # Will raise an error since the name column is missing
    my_function(pd.DataFrame({
        "id": [1, 2, 3],
    })
    ```
    """

    def decorator(func: Callable) -> Callable:
        def func_wrapper(*args, **kwargs) -> Any:  # type: ignore
            from typing import _AnnotatedAlias  # type: ignore

            params_to_check = {
                name: value
                for name, value in func.__annotations__.items()
                if type(value) == _AnnotatedAlias  # noqa: E721
            }

            function_args = func.__code__.co_varnames

            # Naming args variables
            all_args = kwargs.copy()
            for index in range(len(args)):
                all_args[function_args[index]] = args[index]

            def wrapper_metadata(value: Any) -> FeatureViewWrapper | None:
                for val in value.__metadata__:
                    if isinstance(val, FeatureViewWrapper):
                        return val
                return None

            for key, value in params_to_check.items():
                missing_columns = set()

                value = wrapper_metadata(value)
                if value is None:
                    continue

                if key not in all_args:
                    raise ValueError(f"Unable to find {key}")

                view = value.compile()
                df = all_args[key]

                if isinstance(df, (pl.LazyFrame, pl.DataFrame, pd.DataFrame)):
                    columns = df.columns
                elif isinstance(df, dict):
                    columns = list(df.keys())
                else:
                    raise ValueError(f"Invalid data type: {type(df)}")

                for feature in view.request_all.needed_requests[0].all_features:
                    if feature.name not in columns:
                        missing_columns.add(feature.name)

                if missing_columns:
                    raise ValueError(
                        f"Missing columns: {list(missing_columns)} in the dataframe '{key}'\n{df}."
                    )

            return func(*args, **kwargs)

        return func_wrapper

    return decorator
