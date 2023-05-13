from datetime import datetime, timedelta

import pandas as pd
import pytest

from aligned.local.job import FileFullJob
from aligned.retrival_job import DerivedFeatureJob, RetrivalRequest
from aligned.sources.local import LiteralReference


@pytest.mark.asyncio
async def test_derived_pandas(retrival_request_with_derived: RetrivalRequest) -> None:
    date = datetime(year=2022, month=1, day=1)
    one_day = timedelta(days=1)
    job = DerivedFeatureJob(
        job=FileFullJob(
            LiteralReference(
                pd.DataFrame(
                    {
                        'id': [1, 2, 3, 4, 5],
                        'c': [3, 4, 2, 3, 4],
                        'd': [1, 1, 1, 2, 4],
                        'created_at': [date, date, date + one_day, date + one_day, date + one_day],
                    }
                )
            ),
            request=retrival_request_with_derived,
        ),
        requests=[retrival_request_with_derived],
    )

    data = await job.to_pandas()

    assert set(data.columns) == {'id', 'c', 'd', 'created_at', 'c+d'}
    assert data.shape[0] == 5


@pytest.mark.asyncio
async def test_derived_polars(retrival_request_with_derived: RetrivalRequest) -> None:
    date = datetime(year=2022, month=1, day=1)
    one_day = timedelta(days=1)
    job = DerivedFeatureJob(
        job=FileFullJob(
            LiteralReference(
                pd.DataFrame(
                    {
                        'id': [1, 2, 3, 4, 5],
                        'c': [3, 4, 2, 3, 4],
                        'd': [1, 1, 1, 2, 4],
                        'created_at': [date, date, date + one_day, date + one_day, date + one_day],
                    }
                )
            ),
            request=retrival_request_with_derived,
        ),
        requests=[retrival_request_with_derived],
    )

    data = await job.to_pandas()

    assert set(data.columns) == {'id', 'c', 'd', 'created_at', 'c+d'}
    assert data.shape[0] == 5
