import pytest

from aligned.retrival_job import CombineFactualJob, RetrivalJob, RetrivalRequest


@pytest.mark.asyncio
async def test_combined_pandas(
    retrival_job: RetrivalJob,
    retrival_job_with_timestamp: RetrivalJob,
    combined_retrival_request: RetrivalRequest,
) -> None:
    job = CombineFactualJob(
        jobs=[retrival_job, retrival_job_with_timestamp], combined_requests=[combined_retrival_request]
    )
    data = await job.to_pandas()

    assert set(data.columns) == {'id', 'a', 'b', 'c', 'd', 'created_at', 'c+d', 'a+c+d'}
    assert data.shape[0] == 5


@pytest.mark.asyncio
async def test_combined_polars(
    retrival_job: RetrivalJob,
    retrival_job_with_timestamp: RetrivalJob,
    combined_retrival_request: RetrivalRequest,
) -> None:
    job = CombineFactualJob(
        jobs=[retrival_job, retrival_job_with_timestamp], combined_requests=[combined_retrival_request]
    )
    data = (await job.to_lazy_polars()).collect()

    assert set(data.columns) == {'id', 'a', 'b', 'c', 'd', 'created_at', 'c+d', 'a+c+d'}
    assert data.shape[0] == 5
