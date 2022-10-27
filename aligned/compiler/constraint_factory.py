from dataclasses import dataclass

from aligned.schemas.constraints import Constraint
from aligned.schemas.feature_view import CompiledFeatureView


class ConstraintFactory:
    async def compile(self, view: CompiledFeatureView) -> Constraint:
        pass


@dataclass
class LiteralFactory(ConstraintFactory):

    constraint: Constraint

    async def compile(self, view: CompiledFeatureView) -> Constraint:
        return self.constraint
