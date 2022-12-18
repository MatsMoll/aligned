import pytest

from aligned import FeatureStore, FileSource


@pytest.mark.asyncio
async def test_cached_at(titanic_feature_store: FeatureStore) -> None:
    file_source = FileSource.parquet_at('test_data/titanic.parquet')
    data_set = (
        await titanic_feature_store.feature_view('titanic_parquet').all().cached_at(file_source).to_pandas()
    )
    cached = await file_source.read_pandas()

    assert cached.shape[0] == data_set.shape[0]
    assert (set(cached.columns).intersection(set(data_set.columns)) - set(cached.columns)) == set()
