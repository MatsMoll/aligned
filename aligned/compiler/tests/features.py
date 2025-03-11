from dataclasses import dataclass

from pydantic import BaseModel, Field
from aligned import feature_view, UUID, String, PostgreSQLConfig, Float32, Struct, List
from aligned.schemas.feature import FeatureType

source = PostgreSQLConfig.localhost("test")


@feature_view(name="test", description="test", source=source.table("test"))
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


def test_from_type() -> None:
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


def test_struct_as_type() -> None:
    from pyspark.sql.types import StructType, ArrayType

    @feature_view(source=source.table("test"))
    class WithStructure:
        a = UUID().as_entity()

        structure = Struct(BaseModelTest)
        list_with_struct = List(Struct(TestingClass))
        list_of_type = List(Float32())

    req = WithStructure.request
    schema = req.spark_schema()
    assert len(schema.fields) == 4

    all_fields = schema.fieldNames()
    assert all_fields == sorted(all_fields)

    structs = [
        field.dataType
        for field in schema.fields
        if isinstance(field.dataType, StructType)
    ]
    assert len(structs) == 1

    lists = [
        field.dataType
        for field in schema.fields
        if isinstance(field.dataType, ArrayType)
    ]
    assert len(lists) == 2

    lists_with_struct = [
        field.elementType
        for field in lists
        if isinstance(field.elementType, StructType)
    ]
    assert len(lists_with_struct) == 1
