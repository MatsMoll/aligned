import pytest
from aligned import feature_view, String, Int32, InMemorySource


@feature_view(
    source=InMemorySource.from_values(
        {
            "feature_id": ["a", "b", "c"],
            "x": [3, 3, 3],
        }
    )
)
class SomeFeatures:
    feature_id = String().as_entity()

    x = Int32()
    y = Int32().is_optional()

    new_value = x + y
    additional_value = new_value + x


@pytest.mark.asyncio
async def test_partial_features() -> None:
    df = (
        await SomeFeatures.query()
        .features_for({"feature_id": ["a", "b", "c"], "y": [1, 2, 3]})
        .to_polars()
    )

    assert df["new_value"].to_list() == [4, 5, 6]
    assert df["additional_value"].to_list() == [7, 8, 9]
