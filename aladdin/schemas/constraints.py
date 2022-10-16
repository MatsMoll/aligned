from dataclasses import dataclass

from aladdin.schemas.codable import Codable


class Constraint(Codable):
    name: str

    def to_dict(self) -> dict:
        return {'name': self.name}

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Above(Constraint):
    value: float

    name = 'above'

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
class Domain(Constraint):

    values: list[str]
    name = 'domain'

    def __hash__(self) -> int:
        return hash(self.name)
