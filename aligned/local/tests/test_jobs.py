import pandas as pd
import pytest

from aligned.local.job import FileFullJob
from aligned.local.source import LiteralReference
from aligned.retrival_job import RetrivalRequest


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
    data = (await job.to_polars()).collect()

    assert set(data.columns) == {'id', 'a', 'b'}
    assert data.shape[0] == 5
