from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aligned.schemas.feature import Feature, FeatureLocation
from aligned.schemas.vector_storage import VectorIndex, VectorStorage

if TYPE_CHECKING:
    from aligned.compiler.feature_factory import FeatureFactory


@dataclass
class VectorIndexFactory:

    vector_dim: int
    metadata: list[FeatureFactory]
    storage: VectorStorage

    def compile(self, location: FeatureLocation, vector: Feature, entities: set[Feature]) -> VectorIndex:
        return VectorIndex(
            location=location,
            vector=vector,
            vector_dim=self.vector_dim,
            metadata=[feature.feature() for feature in self.metadata],
            storage=self.storage,
            entities=list(entities),
        )
