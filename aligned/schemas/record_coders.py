from dataclasses import dataclass
from typing import Optional

from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable


class RecordCoder(Codable, SerializableType):

    coder_type: str

    def __hash__(self) -> int:
        return hash(self.coder_type)

    def _serialize(self) -> dict:
        assert (
            self.coder_type in SupportedRecordCoders.shared().types
        ), f'RecordCoder {self.coder_type} is not supported'
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> 'RecordCoder':
        name_type = value['coder_type']
        del value['coder_type']
        data_class = SupportedRecordCoders.shared().types[name_type]

        return data_class.from_dict(value)

    def decode(self, records: list[dict]) -> list[dict]:
        pass

    def encode(self, records: list[dict]) -> list[dict]:
        pass


class SupportedRecordCoders:

    types: dict[str, type[RecordCoder]]

    _shared: Optional['SupportedRecordCoders'] = None

    def __init__(self) -> None:
        self.types = {}

        for tran_type in [PassthroughRecordCoder, JsonRecordCoder]:
            self.add(tran_type)

    def add(self, constraint: type[RecordCoder]) -> None:
        self.types[constraint.coder_type] = constraint

    @classmethod
    def shared(cls) -> 'SupportedRecordCoders':
        if not cls._shared:
            cls._shared = SupportedRecordCoders()
        return cls._shared


@dataclass
class PassthroughRecordCoder(RecordCoder):

    coder_type = 'passthrough'

    def decode(self, records: list[dict]) -> list[dict]:
        return records

    def encode(self, records: list[dict]) -> list[dict]:
        return records


@dataclass
class JsonRecordCoder(RecordCoder):

    key: str

    coder_type = 'json'

    def decode(self, records: list[dict]) -> list[dict]:
        import json

        decoded = [json.loads(record[self.key]) for record in records if self.key in record]
        return [record for record in decoded if isinstance(record, dict)]

    def encode(self, records: list[dict]) -> list[dict]:
        import json

        return [{self.key: json.dumps(record)} for record in records]
