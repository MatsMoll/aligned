import pytest

from aligned import feature_view, String, Bool
from aligned.sources.in_mem_source import InMemorySource
from aligned.retrival_job import CombineFactualJob, RetrivalJob, RetrivalRequest
from aligned.compiler.feature_factory import transform_polars, transform_pandas, transform_row

import polars as pl
from aligned.lazy_imports import pandas as pd


@feature_view(source=InMemorySource.empty())
class CombinedData:
    query = String()
    contains_mr = query.contains('mr')

    @transform_polars(using_features=[query], return_type=Bool())
    def contains_something(self, df: pl.LazyFrame, return_value: str) -> pl.LazyFrame:
        return df.with_columns((pl.col('query').str.len_chars() > 5).alias(return_value))

    @transform_pandas(using_features=[query], return_type=String())
    def append_someting(self, df: pd.DataFrame) -> pd.Series:
        return df['query'] + ' something'

    @transform_row(using_features=[query], return_type=String())
    def using_row(self, row: dict) -> str:
        return row['query'] + ' something'

    not_contains = contains_something.not_equals(True)


@pytest.mark.asyncio
async def test_feature_view_without_entity():

    job = CombinedData.query().features_for({'query': ['Hello', 'Hello mr']})
    df = await job.to_polars()

    assert df['contains_mr'].sum() == 1


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
