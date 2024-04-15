import polars as pl
import json
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.request.retrival_request import RetrivalRequest
from aligned.sources.local import CsvFileSource
import pytest


@pytest.mark.asyncio
async def test_custom_transformation_as_lambda(scan_without_datetime: CsvFileSource) -> None:

    new_source = scan_without_datetime.transform_with_polars(
        lambda df: df.with_columns(bucket=pl.col('id').mod(3))
        .groupby('bucket')
        .agg(
            pl.col('radius_mean').sum().alias('sum_radius_mean'),
        )
    )

    df = await new_source.all_data(RetrivalRequest.all_data(), limit=None).to_polars()

    source_as_json = new_source.to_json()

    ds = BatchDataSource._deserialize(json.loads(source_as_json))
    new_df = await ds.all_data(RetrivalRequest.all_data(), limit=None).to_polars()

    assert new_df.sort('bucket').equals(df.sort('bucket').select(new_df.columns))


@pytest.mark.asyncio
async def test_custom_transformation_as_function(scan_without_datetime: CsvFileSource) -> None:
    async def custom_function(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.with_columns(bucket=pl.col('id').mod(3))
            .groupby('bucket')
            .agg(
                pl.col('radius_mean').sum().alias('sum_radius_mean'),
            )
        )

    new_source = scan_without_datetime.transform_with_polars(custom_function)

    df = await new_source.all_data(RetrivalRequest.all_data(), limit=None).to_polars()

    source_as_json = new_source.to_json()

    ds = BatchDataSource._deserialize(json.loads(source_as_json))
    new_df = await ds.all_data(RetrivalRequest.all_data(), limit=None).to_polars()

    assert new_df.sort('bucket').equals(df.sort('bucket').select(new_df.columns))
