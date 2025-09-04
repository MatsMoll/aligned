import logging
from dataclasses import dataclass, field
from datetime import timedelta, datetime

from aligned.request.retrieval_request import (
    EventTimestampRequest,
    FeatureRequest,
    RetrievalRequest,
)
from aligned.schemas.codable import Codable
from aligned.schemas.feature import FeatureLocation, FeatureType, StaticFeatureTags
from aligned.schemas.feature import EventTimestamp, Feature, FeatureReference
from aligned.schemas.event_trigger import EventTrigger
from aligned.schemas.target import (
    ClassificationTarget,
    RecommendationConfig,
    RegressionTarget,
)
from aligned.schemas.feature_view import (
    CompiledFeatureView,
    Contact,
    FeatureViewReferenceSource,
)
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.folder import DatasetStore
from aligned.exposed_model.interface import ExposedModel
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.data_source.batch_data_source import CodableBatchDataSource
from aligned.retrieval_job import RetrievalJob

logger = logging.getLogger(__name__)


@dataclass
class FeatureInputVersions(Codable):
    default_version: str
    versions: dict[str, list[FeatureReference]]

    def features_for(self, version: str) -> list[FeatureReference]:
        return self.versions.get(version, [])

    @property
    def default_features(self) -> list[FeatureReference]:
        return self.features_for(self.default_version)

    def all_features(self) -> set[FeatureReference]:
        all_features = set()
        for version in self.versions.values():
            all_features.update(version)
        return all_features


@dataclass
class Target(Codable):
    estimating: FeatureReference
    feature: Feature

    on_ground_truth_event: StreamDataSource | None = field(default=None)

    # This is a limitation of the current setup.
    # Optimaly will this be on the features feature view, but this is not possible at the moment
    event_trigger: EventTrigger | None = field(default=None)

    def __hash__(self) -> int:
        return self.feature.name.__hash__()


@dataclass
class PredictionsView(Codable):
    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]
    event_timestamp: EventTimestamp | None = field(default=None)

    source: CodableBatchDataSource | None = field(default=None)
    application_source: CodableBatchDataSource | None = field(default=None)
    stream_source: StreamDataSource | None = field(default=None)

    regression_targets: set[RegressionTarget] | None = field(default=None)
    classification_targets: set[ClassificationTarget] | None = field(default=None)
    recommendation_targets: set[RecommendationConfig] | None = field(default=None)

    acceptable_freshness: timedelta | None = field(default=None)
    unacceptable_freshness: timedelta | None = field(default=None)

    @property
    def freshness_feature(self) -> Feature | None:
        if self.event_timestamp:
            return self.event_timestamp.as_feature()

        for feat in self.features:
            if feat.tags and StaticFeatureTags.is_freshness in feat.tags:
                return feat
        return None

    @property
    def is_shadow_model_flag(self) -> Feature | None:
        for feature in self.features:
            if feature.tags and StaticFeatureTags.is_shadow_model in feature.tags:
                return feature
        return None

    @property
    def model_version_column(self) -> Feature | None:
        # Can use annotated by, but is is lower pri then a model version
        annotated_by = None

        for feature in self.features:
            if not feature.tags:
                continue
            if StaticFeatureTags.is_model_version in feature.tags:
                return feature
            if StaticFeatureTags.is_annotated_by in feature.tags:
                annotated_by = feature
        return annotated_by

    @property
    def prompt_completion_feature(self) -> Feature | None:
        for feature in self.features:
            if feature.tags and StaticFeatureTags.is_prompt_completion in feature.tags:
                return feature
        return None

    @property
    def logged_features(self) -> Feature | None:
        for feature in self.features:
            if feature.tags and StaticFeatureTags.is_input_features in feature.tags:
                return feature
        return None

    def as_view(self, name: str) -> CompiledFeatureView:
        from aligned.sources.in_mem_source import InMemorySource
        import polars as pl

        return CompiledFeatureView(
            name=name,
            source=self.source or InMemorySource(pl.DataFrame()),
            entities=self.entities,
            features=self.features,
            derived_features=self.derived_features,
            event_timestamp=self.event_timestamp,
        )

    def schema_hash(self) -> bytes:
        from hashlib import md5

        schema_string = ""

        schema_features = self.features.union(self.entities)
        if self.event_timestamp:
            schema_features.add(self.event_timestamp.as_feature())

        for feature in sorted(schema_features, key=lambda val: val.name):
            if not feature.default_value:
                schema_string += f"{feature.name}:{feature.dtype.name}"

        return md5(schema_string.encode(), usedforsecurity=False).digest()

    @property
    def full_schema(self) -> set[Feature]:
        schema = self.features.union(self.entities).union(self.derived_features)

        for target in self.classification_targets or {}:
            schema.add(target.feature)
            schema.update({prob.feature for prob in target.class_probabilities})

        for target in self.regression_targets or {}:
            schema.add(target.feature)
            if target.confidence:
                schema.add(target.confidence)

            if target.lower_confidence:
                schema.add(target.lower_confidence)

            if target.upper_confidence:
                schema.add(target.upper_confidence)

        if self.model_version_column:
            schema.add(self.model_version_column)

        return schema

    def embeddings(self) -> list[Feature]:
        embeds = [feat for feat in self.full_schema if feat.dtype.is_embedding]
        return sorted(embeds, key=lambda feat: feat.name)

    def request(
        self, name: str, model_version_as_entity: bool = False
    ) -> RetrievalRequest:
        entities = self.entities

        if model_version_as_entity and self.model_version_column:
            entities = entities.union({self.model_version_column})

        event_timestamp = self.event_timestamp
        if not event_timestamp and self.freshness_feature:
            event_timestamp = EventTimestamp(name=self.freshness_feature.name, ttl=None)

        return RetrievalRequest(
            name=name,
            location=FeatureLocation.model(name),
            entities=entities,
            features=self.features,
            derived_features=self.derived_features,
            event_timestamp=event_timestamp,
        )

    def request_for(
        self,
        features: set[str],
        name: str,
        model_version_as_entity: bool = False,
        event_timestamp_column: str | None = None,
    ) -> RetrievalRequest:
        entities = self.entities

        if model_version_as_entity and self.model_version_column:
            entities = entities.union({self.model_version_column})

        event_timestamp_request = None
        if self.event_timestamp:
            event_timestamp_request = EventTimestampRequest(
                event_timestamp=self.event_timestamp,
                entity_column=event_timestamp_column,
            )

        return RetrievalRequest(
            name=name,
            location=FeatureLocation.model(name),
            entities=entities,
            features={feature for feature in self.features if feature.name in features},
            derived_features={
                feature for feature in self.derived_features if feature.name in features
            },
            event_timestamp_request=event_timestamp_request,
        )

    def labels_estimates_refs(self) -> set[FeatureReference]:
        if self.classification_targets:
            return {feature.estimating for feature in self.classification_targets}
        elif self.regression_targets:
            return {feature.estimating for feature in self.regression_targets}
        elif self.recommendation_targets:
            return {
                target.was_selected_list
                for target in self.recommendation_targets
                if target.was_selected_list
            }
        else:
            return set()

    def labels(self) -> set[Feature]:
        if self.classification_targets:
            return {feature.feature for feature in self.classification_targets}
        elif self.regression_targets:
            return {feature.feature for feature in self.regression_targets}
        elif self.recommendation_targets:
            feature_names = {
                feature.feature_name for feature in self.recommendation_targets
            }
            return {feat for feat in self.full_schema if feat.name in feature_names}
        else:
            return {
                feat
                for feat in self.full_schema
                if feat.tags and StaticFeatureTags.is_annotated_feature in feat.tags
            }


@dataclass
class Model(Codable):
    name: str
    features: FeatureInputVersions
    predictions_view: PredictionsView
    description: str | None = field(default=None)
    contacts: list[Contact] | None = field(default=None)
    tags: list[str] | None = field(default=None)
    dataset_store: DatasetStore | None = field(default=None)
    exposed_at_url: str | None = field(default=None)
    exposed_model: ExposedModel | None = field(default=None)

    def __hash__(self) -> int:
        return self.name.__hash__()

    def feature_references(self, version: str | None = None) -> set[FeatureReference]:
        return set(self.features.features_for(version or self.features.default_version))

    @property
    def request_all_predictions(self) -> FeatureRequest:
        return FeatureRequest(
            FeatureLocation.model(self.name),
            {feature.name for feature in self.predictions_view.full_schema},
            needed_requests=[
                RetrievalRequest(
                    name=self.name,
                    location=FeatureLocation.model(self.name),
                    features=self.predictions_view.full_schema,
                    derived_features=self.predictions_view.derived_features,
                    entities=self.predictions_view.entities,
                    event_timestamp=self.predictions_view.event_timestamp,
                )
            ],
        )


@dataclass
class ModelSource(CodableBatchDataSource):
    model: Model
    pred_view: CompiledFeatureView

    type_name: str = "model_source"

    def job_group_key(self) -> str:
        return FeatureLocation.model(self.pred_view.name).identifier

    def location_id(self) -> set[FeatureLocation]:
        return {FeatureLocation.model(self.model.name)}

    async def schema(self) -> dict[str, FeatureType]:
        if self.model.predictions_view.source:
            return await self.model.predictions_view.source.schema()
        return {}

    def source(self) -> FeatureViewReferenceSource:
        return FeatureViewReferenceSource(
            self.pred_view, FeatureLocation.model(self.pred_view.name)
        )

    def all_data(
        self, request: RetrievalRequest, limit: int | None = None
    ) -> RetrievalJob:
        job = self.source().all_data(request, limit)

        model_version = self.model.predictions_view.model_version_column

        if model_version:
            unique_on = [feat.name for feat in self.pred_view.entities]
            return job.unique_on(unique_on)

        return job

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        job = self.source().all_between_dates(request, start_date, end_date)

        model_version = self.model.predictions_view.model_version_column

        if model_version:
            unique_on = [feat.name for feat in self.pred_view.entities]
            return job.unique_on(unique_on)

        return job

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        return self.source().features_for(facts, request)

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls, facts: RetrievalJob, requests: list[tuple["ModelSource", RetrievalRequest]]
    ) -> RetrievalJob:
        if len(requests) != 1:
            raise NotImplementedError(
                f"Type: {cls} have not implemented how to load fact data with multiple sources."
            )

        source, _ = requests[0]
        return source.source().features_for(facts, requests[0][1])

    def depends_on(self) -> set[FeatureLocation]:
        return {FeatureLocation.model(self.pred_view.name)}
