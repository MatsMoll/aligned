import pytest

from aligned.feature_store import FeatureStore
from aligned.schemas.transformation import SupportedTransformations


@pytest.mark.asyncio
async def test_transformations() -> None:
    supported = SupportedTransformations.shared()

    for transformation in supported.types.values():
        await transformation.run_transformation_test()


@pytest.mark.asyncio
async def test_transformations_in_feture_view(alot_of_transforation_feature_store: FeatureStore) -> None:
    store = alot_of_transforation_feature_store

    amount = 100

    data = await store.feature_view('titanic').all(limit=amount).to_df()

    assert data.shape[0] == amount
