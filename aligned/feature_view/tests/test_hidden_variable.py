import pytest

from aligned import (
    Bool,
    Entity,
    Float,
    PostgreSQLConfig,
    String,
    feature_view,
    FileSource,
)
from aligned.feature_view.feature_view import FeatureView, FeatureViewMetadata
from aligned.compiler.feature_factory import compile_hidden_features
from aligned.schemas.feature import FeatureLocation

source = PostgreSQLConfig.localhost('test')


class TestView(FeatureView):

    metadata = FeatureViewMetadata(
        name='test', description='test', tags=['Test'], source=source.table('test')
    )

    test_id = Entity(String())

    variable = String()
    some_bool = Bool()

    is_not_true = (~(variable == 'true')) & some_bool
    is_not_true_other = some_bool & (~(variable == 'true'))
    is_true = variable == 'True'

    y_value = Float()
    x_value = Float()

    some_ratio = (y_value - x_value) / x_value


@pytest.mark.asyncio
async def test_hidden_variable() -> None:

    view = TestView.compile()

    assert len(view.derived_features) == 9


@pytest.mark.asyncio
async def test_select_variables() -> None:

    view = TestView.compile()

    assert len(view.derived_features) == 9

    request = view.request_for({'some_ratio'})

    assert len(request.needed_requests) == 1
    needed_req = request.needed_requests[0]

    assert len(needed_req.features) == 2
    assert len(needed_req.derived_features) == 2


def test_hidden_variable_condition() -> None:
    class Test:
        x, y = Bool(), Bool()
        z = (x & y) | x

    test = Test()

    features, derived_features = compile_hidden_features(
        test.z | test.y,
        FeatureLocation.feature_view('view'),
        hidden_features=0,
        var_name='test',
        entities=set(),
    )

    assert len(features) == 2
    assert len(derived_features) == 3


@pytest.mark.asyncio
async def test_core_feature_as_hidden() -> None:
    @feature_view(name='test', source=FileSource.csv_at('test_data/titanic_dataset.csv'))
    class Test:
        PassengerId = String().as_entity()

        Age = Float().fill_na(10)

    compiled = Test.compile()  # type: ignore
    assert len(compiled.derived_features) == 1

    df = await Test.query().all().to_pandas()  # type: ignore
    assert (~df['Age'].isna()).all()
