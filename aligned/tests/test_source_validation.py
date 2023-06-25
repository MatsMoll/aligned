import pytest

from aligned import FeatureStore, FileSource
from aligned.source_validation import validate_sources_in


@pytest.mark.asyncio
async def test_source_validation(titanic_feature_store: FeatureStore) -> None:

    source = FileSource.parquet_at('test_data/titanic.parquet')

    views = titanic_feature_store.views_with_batch_source(source)
    validation = await validate_sources_in(views)

    assert {'titanic_parquet': True} == validation
