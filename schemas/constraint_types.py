from dataclasses import dataclass
from aligned.schemas.constraints import Constraint
from aligned.schemas.feature import FeatureReference


@dataclass
class ReferencingColumn(Constraint):

    value: FeatureReference
    name = 'referencing_column'

    def __hash__(self) -> int:
        return hash(self.name)
