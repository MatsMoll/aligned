from datetime import timedelta
import pytest
import polars as pl
from aligned import Float32, Int32, List, RedisConfig, String, feature_view, Timestamp
from aligned.sources.random_source import RandomDataSource
from aligned.sources.redis import RedisSource

ttl_duration = timedelta(hours=2)


@feature_view(
    source=RandomDataSource(),
    materialized_source=RedisConfig.from_env(),
    unacceptable_freshness=ttl_duration,
)
class TestView:
    billing_agreement_id = Int32().as_entity()

    x = Int32()
    b = Float32()

    new = x + b

    some_list = List(String())
    list_with_numbers = List(Float32())

    timestamp = Timestamp().as_freshness()


@pytest.mark.asyncio
async def test_redis_writes_and_read() -> None:
    view = TestView.query()

    redis_source = TestView.metadata.materialized_source
    assert redis_source
    assert isinstance(redis_source, RedisSource)

    redis_client = redis_source.config.redis()

    test_data = (
        await view.using_source(TestView.metadata.source).all(limit=10).to_polars()
    )
    array_values = test_data["some_list"].to_list()
    array_values[1] = None

    test_data = test_data.with_columns(
        pl.Series(name="some_list", values=array_values, dtype=pl.List(pl.String()))
    )
    await view.overwrite(test_data)

    id_col = "billing_agreement_id"

    only_entities = test_data.select(id_col)
    features = await view.features_for(only_entities).to_polars()

    assert features.select(test_data.columns).equals(test_data)

    id_list = only_entities[id_col].to_list()
    for entity_id in id_list:
        key = f"feature_view:test_view:{entity_id}"

        ttl = await redis_client.ttl(key)
        assert ttl != -2, "Probably the incorrect key"

        expected_ttl_in_sec = ttl_duration.total_seconds()

        assert ttl <= expected_ttl_in_sec
        assert expected_ttl_in_sec * 0.95 <= ttl

    for i in range(test_data.height + 1):
        if i in id_list:
            continue

        missing_features = await view.features_for({id_col: [i]}).to_polars()

        other_columns = missing_features.columns
        other_columns.remove(id_col)

        # All columns should be null
        assert missing_features.select(
            pl.all_horizontal(
                [pl.col(col_name).is_null() for col_name in other_columns]
            ).alias("res")
        )["res"].to_list()[0]
