from dataclasses import dataclass
from typing import Optional as OptionalType

from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable


class Constraint(Codable, SerializableType):
    name: str

    def description(self) -> str:
        raise NotImplementedError(type(self))

    def __hash__(self) -> int:
        return hash(self.name)

    def _serialize(self) -> dict:
        assert (
            self.name in SupportedConstraints.shared().types
        ), f"Constraint {self.name} is not supported"
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> "Constraint":
        name_type = value["name"]
        del value["name"]
        data_class = SupportedConstraints.shared().types[name_type]

        return data_class.from_dict(value)


class SupportedConstraints:
    types: dict[str, type[Constraint]]

    _shared: OptionalType["SupportedConstraints"] = None

    def __init__(self) -> None:
        from aligned.schemas.constraint_types import (
            ReferencingColumn,
        )

        self.types = {}

        for tran_type in [
            LowerBound,
            LowerBoundInclusive,
            UpperBound,
            UpperBoundInclusive,
            Required,
            Optional,
            InDomain,
            MaxLength,
            MinLength,
            StartsWith,
            EndsWith,
            Unique,
            Regex,
            ReferencingColumn,
            ListConstraint,
        ]:
            self.add(tran_type)

    def add(self, constraint: type[Constraint]) -> None:
        self.types[constraint.name] = constraint

    @classmethod
    def shared(cls) -> "SupportedConstraints":
        if cls._shared:
            return cls._shared
        cls._shared = SupportedConstraints()
        return cls._shared


@dataclass
class LowerBound(Constraint):
    value: float

    name = "lower_bound"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Higher then {self.value}"


@dataclass
class LowerBoundInclusive(Constraint):
    value: float

    name = "lower_bound_inc"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Higher then or includes {self.value}"


@dataclass
class UpperBound(Constraint):
    value: float

    name = "upper_bound"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Lower then {self.value}"


@dataclass
class UpperBoundInclusive(Constraint):
    value: float

    name = "upper_bound_inc"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Lower then or includes {self.value}"


@dataclass
class Unique(Constraint):
    name = "unique"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return "Unique"


@dataclass
class MinLength(Constraint):
    value: int

    name = "min_length"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Minimum length of {self.value}"


@dataclass
class Regex(Constraint):
    value: str

    name = "regex"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Conforms to regex '{self.value}'"


@dataclass
class EndsWith(Constraint):
    value: str

    name = "ends_with"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Ends with '{self.value}'"


@dataclass
class StartsWith(Constraint):
    value: str

    name = "starts_with"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Starts with '{self.value}'"


@dataclass
class MaxLength(Constraint):
    value: int

    name = "max_length"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Max length of '{self.value}'"


@dataclass
class And(Constraint):
    left: Constraint
    right: Constraint
    name = "and"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"{self.left.description()} and {self.right.description()}"


@dataclass
class Required(Constraint):
    name = "required"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return "Is required"


@dataclass
class Optional(Constraint):
    name = "optional"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return "Is optional"


@dataclass
class InDomain(Constraint):
    values: list[str]
    name = "in_domain"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"Is one of the following values {self.values}"


@dataclass
class ListConstraint(Constraint):
    constraints: list[Constraint]
    name = "list"

    def __hash__(self) -> int:
        return hash(self.name)

    def description(self) -> str:
        return f"A list that fulfills the {[ constraint.description() for constraint in self.constraints ]}"
