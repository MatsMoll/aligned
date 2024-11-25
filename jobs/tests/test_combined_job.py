import pytest

from aligned import feature_view, String, Bool
from aligned.compiler.model import model_contract
from aligned.feature_store import ContractStore
from aligned.sources.in_mem_source import InMemorySource
from aligned.retrival_job import CombineFactualJob, RetrivalJob, RetrivalRequest
from aligned.compiler.feature_factory import (
    Embedding,
    List,
    transform_polars,
    transform_pandas,
    transform_row,
)

import polars as pl
from aligned.lazy_imports import pandas as pd


@model_contract(
    name='test_embedding',
    input_features=[],
    output_source=InMemorySource.from_values(
        {
            'vec_id': ['a', 'b', 'c'],
            'value': ['hello there', 'no', 'something else'],
            'embedding': [[1, 2], [1, 0], [0, 9]],
        }
    ),
)
class TestEmbedding:
    vec_id = String().as_entity()
    value = String()
    embedding = Embedding(embedding_size=2)


@feature_view(source=InMemorySource.empty())
class CombinedData:
    query = String()
    contains_mr = query.contains('mr')
    embedding = Embedding(embedding_size=2)

    @transform_polars(using_features=[query], return_type=Bool())
    def contains_something(self, df: pl.LazyFrame, return_value: str, store: ContractStore) -> pl.LazyFrame:
        return df.with_columns((pl.col('query').str.len_chars() > 5).alias(return_value))

    @transform_pandas(using_features=[query], return_type=String())
    def append_someting(self, df: pd.DataFrame, store: ContractStore) -> pd.Series:
        return df['query'] + ' something'

    @transform_row(using_features=[query], return_type=String())
    def using_row(self, row: dict, store: ContractStore) -> str:
        return row['query'] + ' something'

    new_format = String().format_string([query, contains_mr], format='Hello {contains_mr} - {query}')

    @transform_row(using_features=[embedding], return_type=List(String()))
    async def related_entities(self, row: dict, store: ContractStore) -> list[str]:
        df = (
            await store.vector_index('test_embedding')
            .nearest_n_to(entities=[row], number_of_records=2)
            .to_polars()
        )
        return df['vec_id'].to_list()

    not_contains = contains_something.not_equals(True)


@pytest.mark.asyncio
async def test_feature_view_without_entity():
    store = ContractStore.empty()
    store.add_model(TestEmbedding)
    store.add_feature_view(CombinedData)

    job = store.feature_view(CombinedData).features_for(
        {'query': ['Hello', 'Hello mr'], 'embedding': [[1, 3], [0, 10]]}
    )
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
