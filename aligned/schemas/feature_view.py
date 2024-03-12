from __future__ import annotations

from typing import TYPE_CHECKING

from datetime import datetime, timedelta
from dataclasses import dataclass, field


from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import AggregatedFeature, DerivedFeature
from aligned.schemas.event_trigger import EventTrigger
from aligned.schemas.feature import EventTimestamp, Feature, FeatureLocation
from aligned.schemas.vector_storage import VectorIndex

if TYPE_CHECKING:
    from aligned.retrival_job import RetrivalJob


@dataclass
class CompiledFeatureView(Codable):
    name: str
    source: BatchDataSource

    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]

    tags: list[str] | None = field(default=None)
    description: str | None = field(default=None)
    aggregated_features: set[AggregatedFeature] = field(default_factory=set)

    event_timestamp: EventTimestamp | None = field(default=None)
    stream_data_source: StreamDataSource | None = field(default=None)
    application_source: BatchDataSource | None = field(default=None)
    materialized_source: BatchDataSource | None = field(default=None)

    acceptable_freshness: timedelta | None = field(default=None)
    unacceptable_freshness: timedelta | None = field(default=None)

    event_triggers: set[EventTrigger] | None = field(default=None)

    contacts: list[str] | None = field(default=None)
    indexes: list[VectorIndex] | None = field(default=None)

    def __pre_serialize__(self) -> CompiledFeatureView:
        assert isinstance(self.name, str)
        assert isinstance(self.source, BatchDataSource)

        for entity in self.entities:
            assert isinstance(entity, Feature)
        for feature in self.features:
            assert isinstance(feature, Feature)
        for derived_feature in self.derived_features:
            assert isinstance(derived_feature, DerivedFeature)
        for aggregated_feature in self.aggregated_features:
            assert isinstance(aggregated_feature, AggregatedFeature)

        if self.tags is not None:
            assert isinstance(self.tags, list)
        if self.description is not None:
            assert isinstance(self.description, str)
        if self.event_timestamp is not None:
            assert isinstance(self.event_timestamp, EventTimestamp)
        if self.stream_data_source is not None:
            assert isinstance(self.stream_data_source, StreamDataSource)
        if self.application_source is not None:
            assert isinstance(self.application_source, BatchDataSource)
        if self.event_triggers is not None:
            for event_trigger in self.event_triggers:
                assert isinstance(event_trigger, EventTrigger)
        if self.contacts is not None:
            assert isinstance(self.contacts, list)
            for contact in self.contacts:
                assert isinstance(contact, str)
        if self.indexes is not None:
            assert isinstance(self.indexes, list)
            for index in self.indexes:
                assert isinstance(index, VectorIndex)
        return self

    @property
    def full_schema(self) -> set[Feature]:
        return self.entities.union(self.features).union(self.derived_features)

    @property
    def entitiy_names(self) -> set[str]:
        return {entity.name for entity in self.entities}

    @property
    def request_all(self) -> FeatureRequest:
        return FeatureRequest(
            FeatureLocation.feature_view(self.name),
            {feature.name for feature in self.full_schema},
            needed_requests=[
                RetrivalRequest(
                    name=self.name,
                    location=FeatureLocation.feature_view(self.name),
                    entities=self.entities,
                    features=self.features,
                    derived_features=self.derived_features,
                    aggregated_features=self.aggregated_features,
                    event_timestamp=self.event_timestamp,
                )
            ],
        )

    def request_for(self, feature_names: set[str]) -> FeatureRequest:

        features = {feature for feature in self.features if feature.name in feature_names}.union(
            self.entities
        )
        derived_features = {feature for feature in self.derived_features if feature.name in feature_names}
        aggregated_features = {
            feature for feature in self.aggregated_features if feature.name in feature_names
        }
        derived_aggregated_feautres = {feature.derived_feature for feature in self.aggregated_features}

        def dependent_features_for(
            feature: DerivedFeature,
        ) -> tuple[set[Feature], set[Feature], set[AggregatedFeature]]:
            core_features = set()
            derived_features = set()
            aggregated_features = set()

            for dep_ref in feature.depending_on:
                if dep_ref.name == feature.name:
                    continue

                dep_feature = [
                    feat for feat in self.features.union(self.entities) if feat.name == dep_ref.name
                ]
                if len(dep_feature) == 1:
                    core_features.add(dep_feature[0])
                    continue

                dep_features = [
                    feat
                    for feat in self.derived_features.union(derived_aggregated_feautres)
                    if feat.name == dep_ref.name
                ]
                if not dep_features:
                    continue

                dep_feature = dep_features[0]
                if dep_feature in derived_aggregated_feautres:
                    agg_feat = [
                        feat for feat in self.aggregated_features if feat.derived_feature == dep_feature
                    ][0]
                    aggregated_features.add(agg_feat)
                else:
                    derived_features.add(dep_feature)

                core, derived, aggregated = dependent_features_for(dep_feature)
                features.update(core)
                derived_features.update(derived)
                aggregated_features.update(aggregated)

            return core_features, derived_features, aggregated_features

        for dep_feature in derived_features.copy():
            core, intermediate, aggregated = dependent_features_for(dep_feature)
            features.update(core)
            derived_features.update(intermediate)
            aggregated_features.update(aggregated)

        for dep_feature in aggregated_features.copy():
            core, intermediate, aggregated = dependent_features_for(dep_feature)
            features.update(core)
            derived_features.update(intermediate)
            aggregated_features.update(aggregated)

        return FeatureRequest(
            FeatureLocation.feature_view(self.name),
            feature_names,
            needed_requests=[
                RetrivalRequest(
                    name=self.name,
                    location=FeatureLocation.feature_view(self.name),
                    entities=self.entities,
                    features=features - self.entities,
                    derived_features=derived_features,
                    aggregated_features=aggregated_features,
                    event_timestamp=self.event_timestamp,
                    features_to_include=feature_names,
                )
            ],
        )

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        entites = '\n'.join([str(entity) for entity in self.entities])
        input_features = '\n'.join([str(features) for features in self.features])
        transformed_features = '\n'.join([str(features) for features in self.derived_features])
        return f"""
{self.name}
Description: {self.description}
Tags: {self.tags}

Entities:
{entites}

Event Timestamp:
{self.event_timestamp}

Input features:
{input_features}

Transformed features:
{transformed_features}
"""


@dataclass
class CompiledCombinedFeatureView(Codable):
    name: str
    features: set[DerivedFeature]  # FIXME: Should combine this and feature_referances into one class.
    feature_referances: dict[str, list[RetrivalRequest]]
    event_triggers: set[EventTrigger] | None = field(default=None)

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
                if request.location not in requests:
                    requests[request.location] = RetrivalRequest(
                        name=request.name,
                        location=request.location,
                        entities=request.entities,
                        features=set(),
                        derived_features=set(),
                        event_timestamp=request.event_timestamp,
                    )
                requests[request.location].derived_features.update(request.derived_features)
                requests[request.location].features.update(request.features)
                requests[request.location].entities.update(request.entities)

        requests[self.name] = RetrivalRequest(
            name=self.name,
            location=FeatureLocation.combined_view(self.name),
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
                if request.location not in dependent_views:
                    dependent_views[request.location] = RetrivalRequest(
                        name=request.name,
                        location=request.location,
                        entities=request.entities,
                        features=set(),
                        derived_features=set(),
                        aggregated_features=set(),
                        event_timestamp=request.event_timestamp,
                    )
                current = dependent_views[request.location]
                current.derived_features = current.derived_features.union(request.derived_features)
                current.features = current.features.union(request.features)
                current.aggregated_features = current.aggregated_features.union(request.aggregated_features)
                dependent_views[request.location] = current

        dependent_views[self.name] = RetrivalRequest(  # Add the request we want
            name=self.name,
            location=FeatureLocation.combined_view(self.name),
            entities=self.entity_features,
            features=set(),
            derived_features={feature for feature in self.features if feature.name in feature_names},
            aggregated_features=set(),
            event_timestamp=None,
        )

        return FeatureRequest(
            FeatureLocation.combined_view(self.name),
            features_to_include=feature_names,
            needed_requests=list(dependent_views.values()),
        )

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class FeatureViewReferenceSource(BatchDataSource):

    view: CompiledFeatureView
    location: FeatureLocation
    renames: dict[str, str] = field(default_factory=dict)

    type_name = 'view_ref'

    def job_group_key(self) -> str:
        return self.view.name

    def sub_request(self, request: RetrivalRequest) -> RetrivalRequest:

        sub_references: set[str] = request.entity_names.union(request.feature_names)
        if request.event_timestamp:
            sub_references.add(request.event_timestamp.name)

        agg_features = {feat.derived_feature for feat in request.aggregated_features}

        sub_features = self.view.request_all.request_result.all_returned_columns

        for feature in request.derived_features.union(agg_features):
            if feature.name in sub_features:
                sub_references.add(feature.name)

            for depends_on in feature.depending_on:
                if depends_on.name in sub_features:
                    sub_references.add(depends_on.name)

        sub_request = self.view.request_for(sub_references)

        if len(sub_request.needed_requests) != 1:
            raise ValueError('Got mulitple requests for one view. Something odd happend.')

        return sub_request.needed_requests[0]

    @classmethod
    def multi_source_features_for(
        cls: type[FeatureViewReferenceSource],
        facts: RetrivalJob,
        requests: list[tuple[FeatureViewReferenceSource, RetrivalRequest]],
    ) -> RetrivalJob:
        from aligned.local.job import FileFactualJob

        sources = {source.job_group_key() for source, _ in requests if isinstance(source, BatchDataSource)}
        if len(sources) != 1:
            raise NotImplementedError(
                f'Type: {cls} have not implemented how to load fact data with multiple sources.'
            )

        source, request = requests[0]

        sub_source = source.view.materialized_source or source.view.source
        sub_request = source.sub_request(request)

        sub_job = (
            sub_source.all_data(sub_request, limit=None)
            .derive_features()
            .with_request([request])
            .fill_missing_columns()
        )

        if request.aggregated_features:
            available_features = sub_job.aggregate(request).derive_features()
        else:
            available_features = sub_job.derive_features([request])

        return FileFactualJob(available_features, [request], facts).rename(source.renames)

    def all_data(self, request: RetrivalRequest, limit: int | None) -> RetrivalJob:
        sub_source = self.view.materialized_source or self.view.source

        sub_req = self.sub_request(request)

        core_job = sub_source.all_data(sub_req, limit=limit)

        if request.aggregated_features:
            job = core_job.aggregate(request)
        else:
            job = core_job.derive_features().with_request([request]).fill_missing_columns()

        return job.derive_features([request]).rename(self.renames)

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:
        sub_source = self.view.materialized_source or self.view.source

        sub_req = self.sub_request(request)

        core_job = sub_source.all_between_dates(sub_req, start_date, end_date).fill_missing_columns()

        if request.aggregated_features:
            job = core_job.aggregate(request)
        else:
            job = core_job.derive_features().with_request([request]).fill_missing_columns()
        return job.derive_features([request]).rename(self.renames)

    def depends_on(self) -> set[FeatureLocation]:
        return {self.location}

    async def freshness(self, event_timestamp: EventTimestamp) -> datetime | None:
        source = self.view.materialized_source or self.view.source
        return await source.freshness(event_timestamp)
