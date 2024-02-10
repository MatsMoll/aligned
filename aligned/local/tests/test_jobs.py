import pandas as pd
import pytest

from aligned import FeatureStore, FileSource
from aligned.local.job import FileFullJob
from aligned.retrival_job import RetrivalRequest
from aligned.sources.local import LiteralReference


@pytest.mark.asyncio
async def test_file_full_job_pandas(retrival_request_without_derived: RetrivalRequest) -> None:
    frame = pd.DataFrame(
        {
            'id': [1, 2, 3, 4, 5],
            'a': [3, 4, 2, 3, 4],
            'b': [1, 1, 1, 2, 4],
        }
    )
    job = FileFullJob(source=LiteralReference(frame), request=retrival_request_without_derived)
    data = await job.to_pandas()

    assert frame.equals(data)


@pytest.mark.asyncio
async def test_file_full_job_polars(retrival_request_without_derived: RetrivalRequest) -> None:
    frame = pd.DataFrame(
        {
            'id': [1, 2, 3, 4, 5],
            'a': [3, 4, 2, 3, 4],
            'b': [1, 1, 1, 2, 4],
        }
    )
    job = FileFullJob(source=LiteralReference(frame), request=retrival_request_without_derived)
    data = (await job.to_lazy_polars()).collect()

    assert set(data.columns) == {'id', 'a', 'b'}
    assert data.shape[0] == 5


@pytest.mark.asyncio
async def test_write_and_read_feature_store(titanic_feature_store_scd: FeatureStore) -> None:
    source = FileSource.json_at('test_data/feature-store.json')
    definition = titanic_feature_store_scd.repo_definition()
    await source.write(definition.to_json().encode('utf-8'))
    store = await source.feature_store()
    assert store is not None
    assert store.model('titanic').model.predictions_view.acceptable_freshness is not None
