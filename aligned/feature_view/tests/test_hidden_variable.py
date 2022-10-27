import pytest

from aligned import Bool, Entity, FeatureView, FeatureViewMetadata, PostgreSQLConfig, String

source = PostgreSQLConfig.localhost('test')


class TestView(FeatureView):

    metadata = FeatureViewMetadata(name='test', description='test', tags={}, batch_source=source)

    test_id = Entity(String())

    variable = String()
    some_bool = Bool()

    is_not_true = (~(variable == 'true')) & some_bool
    is_not_true_other = some_bool & (~(variable == 'true'))
    is_true = variable == 'True'


@pytest.mark.asyncio
async def test_hidden_variable() -> None:

    view = await TestView.compile()

    assert len(view.derived_features) == 7
