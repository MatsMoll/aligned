from pandas import DataFrame, Series # type: ignore
from aladdin.codable import Codable
from dataclasses import dataclass
from aladdin.feature import FeatureType
from typing import Optional
from mashumaro.types import SerializableType


class Transformation(Codable, SerializableType):
    name: str
    dtype: FeatureType | None # Should be something else

    async def transform(self, df: DataFrame) -> Series:
        pass

    def _serialize(self):
        return self.to_dict()
    
    @classmethod
    def _deserialize(cls, value: dict[str]) -> 'Transformation':
        name_type = value["name"]
        del value["name"]
        data_class = SupportedTransformations.shared().types[name_type]
        return data_class.from_dict(value)



class SupportedTransformations:

    types: dict[str, type[Transformation]]

    _shared: Optional["SupportedTransformations"] = None

    def __init__(self):
        self.types = {}
        from aladdin.feature_types import CustomTransformationV2
        
        for tran_type in [
            Equals,
            CustomTransformationV2
        ]:
            self.add(tran_type)

    def add(self, transformation: type[Transformation]):
        self.types[transformation.name] = transformation

    @classmethod
    def shared(cls) -> "SupportedTransformations":
        if cls._shared:
            return cls._shared
        cls._shared = SupportedTransformations()
        return cls._shared


@dataclass
class Equals(Transformation):

    key: str
    value: str

    name: str = "equals" 
    dtype: FeatureType = FeatureType.bool

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    async def transform(self, df: DataFrame) -> Series:
        return df[self.key] == self.value

