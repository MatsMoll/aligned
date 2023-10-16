import logging
from dataclasses import dataclass, field

from aligned.request.retrival_request import FeatureRequest, RetrivalRequest
from aligned.schemas.codable import Codable
from aligned.schemas.feature import FeatureLocation
from aligned.schemas.feature import EventTimestamp, Feature, FeatureReferance
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.schemas.event_trigger import EventTrigger
from aligned.schemas.target import ClassificationTarget, RegressionTarget
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.schemas.folder import Folder

logger = logging.getLogger(__name__)


@dataclass
class Target(Codable):
    estimating: FeatureReferance
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
    model_version_column: Feature | None = field(default=None)
    event_timestamp: EventTimestamp | None = field(default=None)
    source: BatchDataSource | None = field(default=None)
    stream_source: StreamDataSource | None = field(default=None)

    regression_targets: set[RegressionTarget] | None = field(default=None)
    classification_targets: set[ClassificationTarget] | None = field(default=None)

    @property
    def full_schema(self) -> set[Feature]:

        schema = self.features.union(self.entities)

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

    def request(self, name: str) -> RetrivalRequest:
        entities = self.entities
        if self.model_version_column:
            entities.add(self.model_version_column)
        return RetrivalRequest(
            name=name,
            location=FeatureLocation.model(name),
            entities=entities,
            features=self.features,
            derived_features=self.derived_features,
            event_timestamp=self.event_timestamp,
        )

    def request_for(self, features: set[str], name: str) -> RetrivalRequest:
        entities = self.entities

        # if self.model_version_column:
        #     entities.add(self.model_version_column)

        return RetrivalRequest(
            name=name,
            location=FeatureLocation.model(name),
            entities=entities,
            features={feature for feature in self.features if feature.name in features},
            derived_features={feature for feature in self.derived_features if feature.name in features},
            event_timestamp=self.event_timestamp,
        )

    def labels_estimates_refs(self) -> set[FeatureReferance]:
        if self.classification_targets:
            return {feature.estimating for feature in self.classification_targets}
        elif self.regression_targets:
            return {feature.estimating for feature in self.regression_targets}
        else:
            raise ValueError('Found no targets in the model')

    def labels(self) -> set[Feature]:
        if self.classification_targets:
            return {feature.feature for feature in self.classification_targets}
        elif self.regression_targets:
            return {feature.feature for feature in self.regression_targets}
        else:
            raise ValueError('Found no targets in the model')


@dataclass
class Model(Codable):
    name: str
    features: set[FeatureReferance]
    predictions_view: PredictionsView
    description: str | None = field(default=None)
    contacts: list[str] | None = field(default=None)
    tags: dict[str, str] | None = field(default=None)
    dataset_folder: Folder | None = field(default=None)

    def __hash__(self) -> int:
        return self.name.__hash__()

    @property
    def request_all_predictions(self) -> FeatureRequest:
        return FeatureRequest(
            FeatureLocation.model(self.name),
            {feature.name for feature in self.predictions_view.full_schema},
            needed_requests=[
                RetrivalRequest(
                    name=self.name,
                    location=FeatureLocation.model(self.name),
                    features=self.predictions_view.full_schema,
                    derived_features=self.predictions_view.derived_features,
                    entities=self.predictions_view.entities,
                    event_timestamp=self.predictions_view.event_timestamp,
                )
            ],
        )
