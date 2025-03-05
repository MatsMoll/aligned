from __future__ import annotations

import pytest

from aligned.lazy_imports import pandas as pd
from aligned import ContractStore, FileSource
from aligned.local.job import FileFullJob
from aligned.retrieval_job import RetrievalRequest
from aligned.sources.local import LiteralReference


@pytest.mark.asyncio
async def test_file_full_job_pandas(
    retrieval_request_without_derived: RetrievalRequest,
) -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "a": [3, 4, 2, 3, 4],
            "b": [1, 1, 1, 2, 4],
        }
    )
    job = FileFullJob(
        source=LiteralReference(frame), request=retrieval_request_without_derived
    )
    data = await job.to_pandas()

    assert frame.equals(data)


@pytest.mark.asyncio
async def test_file_full_job_polars(
    retrieval_request_without_derived: RetrievalRequest,
) -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "a": [3, 4, 2, 3, 4],
            "b": [1, 1, 1, 2, 4],
        }
    )
    job = FileFullJob(
        source=LiteralReference(frame), request=retrieval_request_without_derived
    )
    data = await job.to_polars()

    assert set(data.columns) == {"id", "a", "b"}
    assert data.shape[0] == 5


@pytest.mark.asyncio
async def test_write_and_read_feature_store(
    titanic_feature_store_scd: ContractStore,
) -> None:
    source = FileSource.json_at("test_data/temp/feature-store.json")
    definition = titanic_feature_store_scd.repo_definition()
    file = definition.to_json()
    assert isinstance(file, str)
    await source.write(file.encode("utf-8"))
    store = await source.feature_store()
    assert store is not None
    assert (
        store.model("titanic").model.predictions_view.acceptable_freshness is not None
    )
