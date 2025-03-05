from dataclasses import dataclass

from pydantic import BaseModel, Field
from aligned import feature_view, UUID, String, PostgreSQLConfig, Float32
from aligned.schemas.feature import FeatureType

source = PostgreSQLConfig.localhost('test')


@feature_view(name='test', description='test', source=source.table('test'))
class Test:
    id = UUID().as_entity()

    x = String()
    y = Float32()

    z = y**2


@dataclass
class TestingClass:
    a: str
    b: int


class BaseModelTest(BaseModel):
    a: str
    b: int = Field(default=2)


def test_from_type():

    dtype = FeatureType.from_type(TestingClass)
    assert dtype
    assert dtype.is_struct
    assert len(dtype.struct_fields()) == 2

    dtype = FeatureType.from_type(list[int])
    assert dtype
    assert dtype.is_array
    assert dtype.array_subtype() == FeatureType.int64()

    dtype = FeatureType.from_type(BaseModelTest)
    assert dtype
    assert dtype.is_struct
    assert len(dtype.struct_fields()) == 2

    dtype = FeatureType.from_type(Test)
    assert dtype
    assert dtype.is_struct
    assert len(dtype.struct_fields()) == 4
