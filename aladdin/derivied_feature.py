from aladdin.feature import Constraint, Feature, FeatureReferance, FeatureType
from aladdin.transformation import Transformation


class DerivedFeature(Feature):

    depending_on: set[FeatureReferance]
    transformation: Transformation

    def __init__(
        self,
        name: str,
        dtype: FeatureType,
        depending_on: set[FeatureReferance],
        transformation: Transformation,
        description: str | None = None,
        is_target: bool = False,
        tags: dict[str, str] | None = None,
        constraints: set[Constraint] | None = None,
    ):
        self.name = name
        self.dtype = dtype
        self.depending_on = depending_on
        self.transformation = transformation
        self.description = description
        self.is_target = is_target
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
