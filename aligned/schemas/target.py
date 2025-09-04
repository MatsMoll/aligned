from dataclasses import dataclass, field
from typing import Literal

from aligned.data_source.stream_data_source import StreamDataSource
from aligned.schemas.codable import Codable
from aligned.schemas.event_trigger import EventTrigger
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReference
from aligned.schemas.literal_value import LiteralValue


@dataclass
class RecommendationConfig(Codable):
    feature_name: str
    output_type: Literal["rank", "score"]
    item_feature: FeatureReference
    top_k: int

    was_selected_list: FeatureReference | None = field(default=None)
    was_selected_view: FeatureLocation | None = field(default=None)

    def __hash__(self) -> int:
        return self.feature_name.__hash__()


@dataclass
class RegressionTarget(Codable):
    estimating: FeatureReference
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
    estimating: FeatureReference
    feature: Feature

    on_ground_truth_event: StreamDataSource | None = field(default=None)

    event_trigger: EventTrigger | None = field(default=None)

    class_probabilities: set[ClassTargetProbability] = field(default_factory=set)
    confidence: Feature | None = field(default=None)

    def __hash__(self) -> int:
        return self.feature.name.__hash__()
