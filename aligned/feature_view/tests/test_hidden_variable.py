import pytest

from aligned import Bool, Entity, FeatureView, FeatureViewMetadata, Float, PostgreSQLConfig, String

source = PostgreSQLConfig.localhost('test')


class TestView(FeatureView):

    metadata = FeatureViewMetadata(name='test', description='test', tags={}, batch_source=source)

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

    view = await TestView.compile()

    assert len(view.derived_features) == 9


@pytest.mark.asyncio
async def test_select_variables() -> None:

    view = await TestView.compile()

    assert len(view.derived_features) == 9

    request = view.request_for({'some_ratio'})

    assert len(request.needed_requests) == 1
    needed_req = request.needed_requests[0]
    assert len(needed_req.derived_features) == 2


@pytest.mark.asyncio
async def test_equal_result_on_compile() -> None:

    view = await TestView.compile()
    graph_view = TestView.compile_graph_only()

    assert view.entities == graph_view.entities
    assert view.features == graph_view.features
    assert len(view.derived_features) == len(graph_view.derived_features)

    for feature in view.derived_features:
        same_feature = [feat for feat in graph_view.derived_features if feat.name == feature.name][0]
        assert feature.depending_on_names == same_feature.depending_on_names
        assert feature.dtype == same_feature.dtype
