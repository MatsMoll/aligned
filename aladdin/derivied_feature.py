from aladdin.feature import Feature, FeatureType, Constraint
from aladdin.transformation import Transformation

class DerivedFeature(Feature):

    depending_on: set[Feature]
    transformation: Transformation

    def __init__(self, 
        name: str,
        dtype: FeatureType,
        depending_on: set[Feature], 
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

    # @classmethod
    # def __post_deserialize__(cls, obj: "DerivedFeature") -> "DerivedFeature":
    #     obj.constraints = set(obj.constraints) if obj.constraints else None
    #     obj.depending_on = set(obj.depending_on)
    #     return obj