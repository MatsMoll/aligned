import polars as pl
import json
from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    UnknownDataSource,
)
from aligned.request.retrieval_request import RetrievalRequest
from aligned.sources.local import CsvFileSource
import pytest


@pytest.mark.asyncio
async def test_custom_transformation_as_lambda(
    scan_without_datetime: CsvFileSource,
) -> None:
    new_source = scan_without_datetime.transform_with_polars(
        lambda df: df.with_columns(bucket=pl.col("id").mod(3))
        .group_by("bucket")
        .agg(
            pl.col("radius_mean").sum().alias("sum_radius_mean"),
        )
    )

    df = await new_source.all_data(RetrievalRequest.all_data(), limit=None).to_polars()

    source_as_json = new_source.to_json()

    ds = CodableBatchDataSource._deserialize(json.loads(source_as_json))
    new_df = await ds.all_data(RetrievalRequest.all_data(), limit=None).to_polars()

    assert new_df.sort("bucket").equals(df.sort("bucket").select(new_df.columns))


async def custom_function(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.with_columns(bucket=pl.col("id").mod(3))
        .group_by("bucket")
        .agg(
            pl.col("radius_mean").sum().alias("sum_radius_mean"),
        )
    )


@pytest.mark.asyncio
async def test_custom_transformation_as_function(
    scan_without_datetime: CsvFileSource,
) -> None:
    new_source = scan_without_datetime.transform_with_polars(custom_function)

    df = await new_source.all_data(RetrievalRequest.all_data(), limit=None).to_polars()

    source_as_json = new_source.to_json()

    ds = CodableBatchDataSource._deserialize(json.loads(source_as_json))
    new_df = await ds.all_data(RetrievalRequest.all_data(), limit=None).to_polars()

    assert new_df.sort("bucket").equals(df.sort("bucket").select(new_df.columns))


def test_decode_unknown_source():
    content = {"type_name": "some-random-source", "some_random_prop": "foo"}
    source = CodableBatchDataSource._deserialize(content)
    assert isinstance(source, UnknownDataSource)

    dict_vals = source.to_dict()
    assert len(dict_vals) == len(content)
