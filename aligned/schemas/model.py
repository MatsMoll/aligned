import logging
from dataclasses import dataclass, field

from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.event_trigger import EventTrigger
from aligned.schemas.feature import EventTimestamp, Feature, FeatureReferance
from aligned.schemas.literal_value import LiteralValue

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
class TargetProbability(Codable):
    outcome: LiteralValue
    feature: Feature


@dataclass
class PredictionsView(Codable):
    target: set[Target]
    entities: set[Feature]
    features: set[Feature]
    derived_features: set[DerivedFeature]
    event_timestamp: EventTimestamp | None = field(default=None)
    source: BatchDataSource | None = field(default=None)
    probabilities: set[TargetProbability] | None = field(default=None)


@dataclass
class Model(Codable):
    name: str
    features: set[FeatureReferance]
    predictions_view: PredictionsView
    description: str | None = field(default=None)

    def __hash__(self) -> int:
        return self.name.__hash__()
