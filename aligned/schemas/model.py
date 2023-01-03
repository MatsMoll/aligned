from dataclasses import dataclass, field

from aligned.schemas.codable import Codable
from aligned.schemas.feature import FeatureReferance


@dataclass
class Model(Codable):
    name: str
    features: set[FeatureReferance]
    targets: set[FeatureReferance] | None = field(default=None)
    model_register: str | None = field(default=None)

    def __hash__(self) -> int:
        return self.name.__hash__()
