import pytest

from aligned import (
    Bool,
    Float32,
    PostgreSQLConfig,
    String,
    feature_view,
    FileSource,
    Int16,
)
from aligned.compiler.feature_factory import compile_hidden_features
from aligned.schemas.feature import FeatureLocation
from aligned.sources.in_mem_source import InMemorySource

source = PostgreSQLConfig.localhost("test")


@feature_view(
    name="test", description="test", tags=["Test"], source=source.table("test")
)
class TestView:
    test_id = String().as_entity()

    variable = String()
    some_bool = Bool()

    is_not_true = (~(variable == "true")) & some_bool
    is_not_true_other = some_bool & (~(variable == "true"))
    is_true = variable == "True"

    y_value = Float32()
    x_value = Float32()

    some_ratio = (y_value - x_value) / x_value


@pytest.mark.asyncio
async def test_hidden_variable() -> None:
    view = TestView.compile()

    assert len(view.derived_features) == 4


@pytest.mark.asyncio
async def test_select_variables() -> None:
    view = TestView.compile()

    assert len(view.derived_features) == 4

    request = view.request_for({"some_ratio"})

    assert len(request.needed_requests) == 1
    needed_req = request.needed_requests[0]

    assert len(needed_req.features) == 2
    assert len(needed_req.derived_features) == 1


def test_hidden_variable_condition() -> None:
    class Test:
        x, y = Bool(), Bool()
        z = (x & y) | x

    test = Test()

    features, derived_features = compile_hidden_features(
        test.z | test.y,
        FeatureLocation.feature_view("view"),
        hidden_features=0,
        var_name="test",
        entities=set(),
    )

    assert len(features) == 2
    assert len(derived_features) == 3


@pytest.mark.asyncio
async def test_hidden_feature_lookup() -> None:
    @feature_view(
        source=InMemorySource.from_values(
            {"some_id": [1, 2], "x": [True, False], "y": [False, False]}
        )
    )
    class Test:
        some_id = Int16().as_entity()

        x, y = Bool(), Bool()
        z = (x & y) | x

        other_value = (some_id > 10) | (some_id.is_in([1, -1])) | (some_id < -10)

    store = Test.query()

    df = await store.features_for({"some_id": [1]}).to_polars()
    assert df.height == 1
    df = await store.select(["z"]).features_for({"some_id": [1]}).to_polars()


@pytest.mark.asyncio
async def test_core_feature_as_hidden() -> None:
    @feature_view(
        name="test", source=FileSource.csv_at("test_data/titanic_dataset.csv")
    )
    class Test:
        PassengerId = String().as_entity()

        Age = Float32().fill_na(10)

    compiled = Test.compile()  # type: ignore
    assert len(compiled.derived_features) == 1

    df = await Test.query().all().to_pandas()  # type: ignore
    assert (~df["Age"].isna()).all()  # type: ignore
