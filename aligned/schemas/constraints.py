from dataclasses import dataclass
from typing import Optional

from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable


class Constraint(Codable, SerializableType):
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def _serialize(self) -> dict:
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> 'Constraint':
        name_type = value['name']
        del value['name']
        data_class = SupportedConstraints.shared().types[name_type]

        return data_class.from_dict(value)


class SupportedConstraints:

    types: dict[str, type[Constraint]]

    _shared: Optional['SupportedConstraints'] = None

    def __init__(self) -> None:
        self.types = {}

        for tran_type in [
            LowerBound,
            LowerBoundInclusive,
            UpperBound,
            UpperBoundInclusive,
            Required,
            InDomain,
        ]:
            self.add(tran_type)

    def add(self, constraint: type[Constraint]) -> None:
        self.types[constraint.name] = constraint

    @classmethod
    def shared(cls) -> 'SupportedConstraints':
        if cls._shared:
            return cls._shared
        cls._shared = SupportedConstraints()
        return cls._shared


@dataclass
class LowerBound(Constraint):
    value: float

    name = 'lower_bound'

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class LowerBoundInclusive(Constraint):
    value: float

    name = 'lower_bound_inc'

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class UpperBound(Constraint):
    value: float

    name = 'upper_bound'

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class UpperBoundInclusive(Constraint):
    value: float

    name = 'upper_bound_inc'

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class And(Constraint):

    left: Constraint
    right: Constraint
    name = 'and'

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Required(Constraint):

    name = 'requierd'

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class InDomain(Constraint):

    values: list[str]
    name = 'in_domain'

    def __hash__(self) -> int:
        return hash(self.name)
