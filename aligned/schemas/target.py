from dataclasses import dataclass, field

from aligned.data_source.stream_data_source import StreamDataSource
from aligned.schemas.codable import Codable
from aligned.schemas.event_trigger import EventTrigger
from aligned.schemas.feature import Feature, FeatureReferance
from aligned.schemas.literal_value import LiteralValue


@dataclass
class RegressionTarget(Codable):
    estimating: FeatureReferance
    feature: Feature

    on_ground_truth_event: StreamDataSource | None = field(default=None)

    event_trigger: EventTrigger | None = field(default=None)

    confidence: Feature | None = field(default=None)
    lower_confidence: Feature | None = field(default=None)
    upper_confidence: Feature | None = field(default=None)

    def __hash__(self) -> int:
        return self.feature.name.__hash__()


@dataclass
class ClassTargetProbability(Codable):
    outcome: LiteralValue
    feature: Feature

    def __hash__(self) -> int:
        return self.feature.name.__hash__()


@dataclass
class ClassificationTarget(Codable):
    estimating: FeatureReferance
    feature: Feature

    on_ground_truth_event: StreamDataSource | None = field(default=None)

    event_trigger: EventTrigger | None = field(default=None)

    class_probabilities: set[ClassTargetProbability] = field(default_factory=set)
    confidence: Feature | None = field(default=None)

    def __hash__(self) -> int:
        return self.feature.name.__hash__()


@dataclass
class RecommendationTarget(Codable):

    estimating: FeatureReferance
    feature: Feature

    estimating_rank: FeatureReferance | None = field(default=None)

    def __hash__(self) -> int:
        return self.feature.name.__hash__()
