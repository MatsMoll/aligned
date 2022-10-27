from aligned.schemas.feature import Constraint, Feature, FeatureReferance, FeatureType
from aligned.schemas.transformation import Transformation


class DerivedFeature(Feature):

    depending_on: set[FeatureReferance]
    transformation: Transformation
    depth: int = 1

    def __init__(
        self,
        name: str,
        dtype: FeatureType,
        depending_on: set[FeatureReferance],
        transformation: Transformation,
        depth: int,
        description: str | None = None,
        tags: dict[str, str] | None = None,
        constraints: set[Constraint] | None = None,
    ):
        self.name = name
        self.dtype = dtype
        self.depending_on = depending_on
        self.transformation = transformation
        self.depth = depth
        self.description = description
        self.tags = tags
        self.constraints = constraints

    @property
    def depending_on_names(self) -> list[str]:
        return [feature.name for feature in self.depending_on]

    @property
    def depending_on_views(self) -> set[str]:
        return {feature.feature_view for feature in self.depending_on}

    @property
    def feature(self) -> Feature:
        return Feature(
            name=self.name,
            dtype=self.dtype,
            description=self.description,
            tags=self.tags,
            constraints=self.constraints,
        )
