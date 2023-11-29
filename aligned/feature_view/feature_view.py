from __future__ import annotations

import copy
import logging

from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, Generic, Type, Callable
from uuid import uuid4

from aligned.compiler.feature_factory import (
    AggregationTransformationFactory,
    Embedding,
    Entity,
    EventTimestamp,
    Bool,
)
from aligned.data_source.batch_data_source import (
    BatchDataSource,
    JoinAsofDataSource,
    JoinDataSource,
    join_asof_source,
    join_source,
    resolve_keys,
)
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.schemas.derivied_feature import (
    AggregatedFeature,
)
from aligned.schemas.feature import FeatureLocation, FeatureReferance
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.compiler.feature_factory import FeatureFactory

if TYPE_CHECKING:
    from aligned.feature_store import FeatureViewStore
    from datetime import datetime

# Enables code compleation in the select method
T = TypeVar('T')


logger = logging.getLogger(__name__)


@dataclass
class FeatureViewMetadata:
    name: str
    source: BatchDataSource
    description: str | None = field(default=None)
    stream_source: StreamDataSource | None = field(default=None)
    application_source: BatchDataSource | None = field(default=None)
    materialized_source: BatchDataSource | None = field(default=None)
    contacts: list[str] | None = field(default=None)
    tags: dict[str, str] = field(default_factory=dict)

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
        )


def feature_view(
    name: str,
    source: BatchDataSource | FeatureViewWrapper,
    description: str | None = None,
    stream_source: StreamDataSource | None = None,
    application_source: BatchDataSource | None = None,
    materialized_source: BatchDataSource | None = None,
    contacts: list[str] | None = None,
    tags: dict[str, str] | None = None,
) -> Callable[[Type[T]], FeatureViewWrapper[T]]:
    def decorator(cls: Type[T]) -> FeatureViewWrapper[T]:

        if isinstance(source, FeatureViewWrapper):
            from aligned.schemas.feature_view import FeatureViewReferenceSource

            compiled = source.compile()
            used_source = FeatureViewReferenceSource(compiled)
        elif isinstance(source, BatchDataSource):
            used_source = source
        else:
            raise ValueError(f'Unable to use source: {source}')

        metadata = FeatureViewMetadata(
            name,
            used_source,
            description=description,
            stream_source=stream_source,
            application_source=application_source,
            materialized_source=materialized_source,
            contacts=contacts,
            tags=tags or {},
        )
        return FeatureViewWrapper(metadata, cls)

    return decorator


@dataclass
class FeatureViewWrapper(Generic[T]):

    metadata: FeatureViewMetadata
    view: Type[T]

    def __call__(self) -> T:
        # Needs to compiile the model to set the location for the view features
        _ = self.compile()
        view = copy.deepcopy(self.view())
        setattr(view, '__view_wrapper__', self)

        for attribute in dir(view):
            if attribute.startswith('__'):
                continue

            value = getattr(view, attribute)
            if isinstance(value, FeatureFactory):
                value._location = FeatureLocation.feature_view(self.metadata.name)
                setattr(view, attribute, copy.deepcopy(value))

        return view

    def compile(self) -> CompiledFeatureView:

        return FeatureView.compile_with_metadata(self.view(), self.metadata)

    def filter(
        self, name: str, where: Callable[[T], Bool], materialize_source: BatchDataSource | None = None
    ) -> FeatureViewWrapper[T]:

        from aligned.data_source.batch_data_source import FilteredDataSource
        from aligned.schemas.feature_view import FeatureViewReferenceSource

        meta = copy.deepcopy(self.metadata)
        meta.name = name

        condition = where(self.__call__())

        main_source = FeatureViewReferenceSource(self.compile())

        if not condition._name:
            condition._name = str(uuid4())
            condition._location = FeatureLocation.feature_view(name)

        if condition.transformation:
            meta.source = FilteredDataSource(main_source, condition.compile())
        else:
            meta.source = FilteredDataSource(main_source, condition.feature())

        if materialize_source:
            meta.materialized_source = materialize_source
            meta.source = main_source

        return FeatureViewWrapper(metadata=meta, view=self.view)

    def join(
        self, view: Any, on: str | FeatureFactory | list[str] | list[FeatureFactory], how: str = 'inner'
    ) -> JoinDataSource:
        from aligned.schemas.feature_view import FeatureViewReferenceSource

        compiled_view = self.compile()
        source = FeatureViewReferenceSource(compiled_view)

        return join_source(source, view, on, how, left_request=compiled_view.request_all.needed_requests[0])

    def join_asof(
        self, view: Any, on: str | FeatureFactory | list[str] | list[FeatureFactory]
    ) -> JoinAsofDataSource:
        from aligned.schemas.feature_view import FeatureViewReferenceSource

        compiled_view = self.compile()
        source = FeatureViewReferenceSource(compiled_view)

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

    def with_entity_renaming(self, named: str, renames: dict[str, str] | str) -> FeatureViewWrapper[T]:
        from aligned.data_source.batch_data_source import ColumnFeatureMappable
        import copy

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
                    'is impossible. Need to setup a dict to know which entity to rename.'
                )

            entity_name = list(compiled_view.entitiy_names)[0]
            renames = {entity_name: renames}

        for source in all_data_sources:
            if not isinstance(source, ColumnFeatureMappable):
                logger.info(
                    f'Source {type(source)} do not conform to ColumnFeatureMappable,'
                    'which could lead to problems'
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
            "id": [1, 2, 3],
        }).to_pandas()
        ```

        Returns:
            FeatureViewStore: Returns a queryable `FeatureViewStore` containing the feature view
        """
        from aligned import FeatureStore

        store = FeatureStore.experimental()
        store.add_compiled_view(self.compile())
        return store.feature_view(self.metadata.name)

    async def process(self, data: dict[str, list[Any]]) -> list[dict]:
        df = await self.query().process_input(data).to_polars()
        return df.collect().to_dicts()

    async def freshness_in_source(self, source: BatchDataSource) -> datetime | None:
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
        batch_source: BatchDataSource,
        description: str | None = None,
        stream_source: StreamDataSource | None = None,
        application_source: BatchDataSource | None = None,
        staging_source: BatchDataSource | None = None,
        contacts: list[str] | None = None,
        tags: dict[str, str] | None = None,
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
            tags=tags or {},
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
    async def freshness_in_source(view: CompiledFeatureView, source: BatchDataSource) -> datetime | None:
        if not view.event_timestamp:
            raise ValueError(
                f'The feature view: {view.name}, needs an event timestamp',
                'to compute the freshness of a source',
            )
        return await source.freshness(view.event_timestamp)

    @staticmethod
    def compile_with_metadata(feature_view: Any, metadata: FeatureViewMetadata) -> CompiledFeatureView:
        from aligned.compiler.feature_factory import FeatureFactory

        # Used to deterministicly init names for hidden features
        hidden_features = 0

        var_names = [name for name in feature_view.__dir__() if not name.startswith('_')]

        view = CompiledFeatureView(
            name=metadata.name,
            description=metadata.description,
            tags=metadata.tags,
            source=metadata.source,
            entities=set(),
            features=set(),
            derived_features=set(),
            aggregated_features=set(),
            event_timestamp=None,
            stream_data_source=metadata.stream_source,
            application_source=metadata.application_source,
            materialized_source=metadata.materialized_source,
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
                        index.compile(feature._location, compiled_feature, view.entities)
                        for index in feature.indexes
                    ]
                )

            if feature.transformation:
                # Adding features that is not stored in the view
                # e.g:
                # class SomeView(FeatureView):
                #     ...
                #     x, y = Bool(), Bool()
                #     z = (x & y) | x
                #
                # Here will (x & y)'s result be a 'hidden' feature
                feature_deps = [(feat.depth(), feat) for feat in feature.feature_dependencies()]

                # Sorting by key in order to instanciate the "core" features first
                # And then making it possible for other features to reference them
                def sort_key(x: tuple[int, FeatureFactory]) -> int:
                    return x[0]

                for depth, feature_dep in sorted(feature_deps, key=sort_key):

                    if not feature_dep._location:
                        feature_dep._location = FeatureLocation.feature_view(metadata.name)
                    elif feature_dep._location.name != metadata.name:
                        continue

                    if feature_dep._name:
                        feat_dep = feature_dep.feature()
                        if feat_dep in view.features or feat_dep in view.entities:
                            continue

                    if depth == 0:
                        # The raw value and the transformed have the same name
                        feature_dep._name = var_name
                        feat_dep = feature_dep.feature()
                        view.features.add(feat_dep)
                        continue

                    if not feature_dep._name:
                        feature_dep._name = str(hidden_features)
                        hidden_features += 1

                    if isinstance(feature_dep.transformation, AggregationTransformationFactory):
                        aggregations.append(feature_dep)
                    else:
                        feature_graph = feature_dep.compile()  # Should decide on which payload to send
                        if feature_graph in view.derived_features:
                            continue

                        view.derived_features.add(feature_dep.compile())

                if isinstance(feature.transformation, AggregationTransformationFactory):
                    aggregations.append(feature)
                else:
                    view.derived_features.add(feature.compile())  # Should decide on which payload to send

            elif isinstance(feature, Entity):
                view.entities.add(compiled_feature)
            elif isinstance(feature, EventTimestamp):
                if view.event_timestamp is not None:
                    raise Exception(
                        'Can only have one EventTimestamp for each'
                        ' FeatureViewDefinition. Check that this is the case for'
                        f' {type(view).__name__}'
                    )
                view.features.add(compiled_feature)
                view.event_timestamp = feature.event_timestamp()
            else:
                view.features.add(compiled_feature)

        if not view.entities:
            raise ValueError(f'FeatureView {metadata.name} must contain at least one Entity')

        loc = FeatureLocation.feature_view(view.name)
        aggregation_group_by = [FeatureReferance(entity.name, loc, entity.dtype) for entity in view.entities]
        event_timestamp_ref = (
            FeatureReferance(view.event_timestamp.name, loc, view.event_timestamp.dtype)
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
        from aligned import FeatureStore

        self = cls()
        store = FeatureStore.experimental()
        store.add_feature_view(self)
        return store.feature_view(self.metadata.name)

    @classmethod
    async def process(cls, data: dict[str, list[Any]]) -> list[dict]:
        df = await cls.query().process_input(data).to_polars()
        return df.collect().to_dicts()

    @staticmethod
    def feature_view_code_template(
        schema: dict[str, FeatureFactory], batch_source_code: str, view_name: str, imports: str | None = None
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
        feature_code = ''
        for name, dtype in schema.items():
            type_name = dtype.__class__.__name__
            data_types.add(type_name)
            feature_code += f'{name} = {type_name}()\n    '

        all_types = ', '.join(data_types)

        return f"""
from aligned import feature_view, {all_types}
{imports or ""}

@feature_view(
    name="{view_name}",
    description="some description",
    source={batch_source_code}
    stream_source=None,
)
class MyView:

    {feature_code}
    """
