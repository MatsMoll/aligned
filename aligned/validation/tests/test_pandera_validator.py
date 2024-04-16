import pytest

from aligned import ContractStore
from aligned.validation.pandera import PanderaValidator


@pytest.mark.asyncio
async def test_validate_valid_feature_view(titanic_feature_store: ContractStore) -> None:
    original = await titanic_feature_store.feature_view('titanic').all(limit=5).to_pandas()
    validated_df = (
        await titanic_feature_store.feature_view('titanic')
        .all(limit=5)
        .drop_invalid(PanderaValidator())
        .to_pandas()
    )

    assert original.shape == validated_df.shape


@pytest.mark.asyncio
async def test_validate_invalid_feature_view(titanic_feature_store: ContractStore) -> None:
    validated_df = (
        await titanic_feature_store.feature_view('titanic')
        .all(limit=20)
        .drop_invalid(PanderaValidator())
        .to_pandas()
    )

    assert validated_df.shape[0] == 16


@pytest.mark.asyncio
async def test_return_invalid_rows(titanic_feature_store: ContractStore) -> None:
    validated_job = titanic_feature_store.feature_view('titanic').all(limit=20).return_invalid()

    validated_df = await validated_job.to_pandas()

    assert validated_df.shape[0] == 4
    assert validated_df.shape[1] == 11

    with_validation = await (
        titanic_feature_store.feature_view('titanic')
        .all(limit=20)
        .return_invalid(should_return_validation=True)
        .to_polars()
    )
    assert with_validation.shape[0] == 4
    assert with_validation.shape[1] == 20
