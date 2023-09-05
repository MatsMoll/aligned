from __future__ import annotations

from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, Generic, Type, Callable

from aligned.compiler.feature_factory import (
    AggregationTransformationFactory,
    Embedding,
    Entity,
    EventTimestamp,
)
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.schemas.derivied_feature import AggregatedFeature, AggregateOver, AggregationTimeWindow
from aligned.schemas.feature import FeatureLocation, FeatureReferance
from aligned.schemas.feature_view import CompiledFeatureView

if TYPE_CHECKING:
    from aligned.compiler.feature_factory import FeatureFactory
    from aligned.feature_store import FeatureViewStore
    from datetime import datetime

# Enables code compleation in the select method
T = TypeVar('T')


@dataclass
class FeatureViewMetadata:
    name: str
    batch_source: BatchDataSource
    description: str | None = field(default=None)
    stream_source: StreamDataSource | None = field(default=None)
    application_source: BatchDataSource | None = field(default=None)
    staging_source: BatchDataSource | None = field(default=None)
    contacts: list[str] | None = field(default=None)
    tags: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_compiled(view: CompiledFeatureView) -> FeatureViewMetadata:
        return FeatureViewMetadata(
            name=view.name,
            description=view.description,
            tags=view.tags,
            batch_source=view.batch_data_source,
            stream_source=view.stream_data_source,
            application_source=view.application_source,
            staging_source=view.staging_source,
        )


def feature_view(
    name: str,
    batch_source: BatchDataSource,
    description: str | None = None,
    stream_source: StreamDataSource | None = None,
    application_source: BatchDataSource | None = None,
    staging_source: BatchDataSource | None = None,
    contacts: list[str] | None = None,
    tags: dict[str, str] | None = None,
) -> Callable[[Type[T]], FeatureViewWrapper[T]]:
    def decorator(cls: Type[T]) -> FeatureViewWrapper[T]:
        metadata = FeatureViewMetadata(
            name,
            batch_source,
            description=description,
            stream_source=stream_source,
            application_source=application_source,
            staging_source=staging_source,
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
        return self.view()

    def compile(self) -> CompiledFeatureView:

        return FeatureView.compile_with_metadata(self.view(), self.metadata)

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
        return await FeatureView.freshness_in_source(compiled, compiled.batch_data_source)


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
            staging_source=staging_source,
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
        return await FeatureView.freshness_in_source(compiled, compiled.batch_data_source)

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
            batch_data_source=metadata.batch_source,
            entities=set(),
            features=set(),
            derived_features=set(),
            aggregated_features=set(),
            event_timestamp=None,
            stream_data_source=metadata.stream_source,
            application_source=metadata.application_source,
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

        aggregation_group_by = [
            FeatureReferance(entity.name, FeatureLocation.feature_view(view.name), entity.dtype)
            for entity in view.entities
        ]

        for aggr in aggregations:
            agg_trans = aggr.transformation
            if not isinstance(agg_trans, AggregationTransformationFactory):
                continue

            if view.event_timestamp is None and agg_trans.time_window:
                raise ValueError(f'FeatureView {metadata.name} must contain an EventTimestamp')

            time_window: AggregationTimeWindow | None = None
            if agg_trans.time_window:

                timestamp_ref = FeatureReferance(
                    view.event_timestamp.name,
                    FeatureLocation.feature_view(view.name),
                    dtype=view.event_timestamp.dtype,
                )
                time_window = AggregationTimeWindow(agg_trans.time_window, timestamp_ref)

            aggr.transformation = agg_trans.with_group_by(aggregation_group_by)
            config = AggregateOver(aggregation_group_by, window=time_window, condition=None)
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

        >>> \"\"\"from aligned import FeatureView, String, Int64, Float

        class MyView(FeatureView):

            metadata = FeatureView.metadata_with(
                name="titanic",
                description="some description",
                batch_source=FileSource.parquest("my_path.parquet")
                stream_source=None,
            )

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
from aligned import FeatureView, {all_types}
{imports or ""}

class MyView(FeatureView):

    metadata = FeatureView.metadata_with(
        name="{view_name}",
        description="some description",
        batch_source={batch_source_code}
        stream_source=None,
    )

    {feature_code}
    """
