from dataclasses import dataclass

from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.schemas.constraints import Constraint


class ConstraintFactory:
    async def compile(self, view: CompiledFeatureView) -> Constraint:
        pass


@dataclass
class LiteralFactory(ConstraintFactory):

    constraint: Constraint

    async def compile(self, view: CompiledFeatureView) -> Constraint:
        return self.constraint
