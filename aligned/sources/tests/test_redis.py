import pytest
from aligned import Float32, Int32, List, RedisConfig, String, feature_view
from aligned.sources.random_source import RandomDataSource


@feature_view(source=RandomDataSource(), materialized_source=RedisConfig.from_env())
class TestView:
    billing_agreement_id = Int32().as_entity()

    x = Int32()
    b = Float32()

    new = x + b

    some_list = List(String())
    list_with_numbers = List(Float32())


@pytest.mark.asyncio
async def test_redis_writes_and_read() -> None:
    view = TestView.query()

    test_data = (
        await view.using_source(TestView.metadata.source).all(limit=10).to_polars()
    )

    await view.overwrite(test_data)

    only_entities = test_data.select("billing_agreement_id")
    features = await view.features_for(only_entities).to_polars()

    assert features.select(test_data.columns).equals(test_data)
