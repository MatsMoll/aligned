from datetime import timedelta

import pytest

from aladdin import FileSource


@pytest.mark.asyncio
async def test_cache_enricher(mocker) -> None:  # type: ignore
    cache_time = timedelta(hours=1)
    source = FileSource(path='test_data/data-with-datetime.csv', mapping_keys={}).enricher()
    enricher = source.cache(ttl=cache_time, cache_key='cached_data')

    file = await enricher.load()
    expected = await source.load()
    assert file.equals(expected)

    pandas_mock = mocker.patch('pandas.read_parquet', return_value=file)
    new_file = await enricher.load()

    assert file.equals(new_file)
    pandas_mock.assert_called_once()
