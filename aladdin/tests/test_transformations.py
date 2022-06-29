import pytest

from aladdin.transformation import SupportedTransformations


@pytest.mark.asyncio
async def test_transformations() -> None:
    supported = SupportedTransformations.shared()

    for transformation in supported.types.values():
        await transformation.run_transformation_test()
