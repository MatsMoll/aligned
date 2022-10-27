import pytest

from aligned import FeatureStore
from aligned.validation.pandera import PanderaValidator


@pytest.mark.asyncio
async def test_validate_valid_feature_view(titanic_feature_store: FeatureStore) -> None:
    original = await titanic_feature_store.feature_view('titanic').all(limit=5).to_df()
    validated_df = (
        await titanic_feature_store.feature_view('titanic').all(limit=5).validate(PanderaValidator()).to_df()
    )

    assert original.shape == validated_df.shape


@pytest.mark.asyncio
async def test_validate_invalid_feature_view(titanic_feature_store: FeatureStore) -> None:
    validated_df = (
        await titanic_feature_store.feature_view('titanic').all(limit=20).validate(PanderaValidator()).to_df()
    )

    assert validated_df.shape[0] == 16
