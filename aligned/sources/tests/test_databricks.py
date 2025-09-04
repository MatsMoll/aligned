import polars as pl
from aligned.schemas.feature import FeatureLocation
from aligned.sources.databricks import (
    DatabricksConnectionConfig,
    UCSqlJob,
    UnityCatalogTableAllJob,
)

config = DatabricksConnectionConfig.serverless(host="test", token="invalid")

sql_source = config.sql("SELECT * FROM some_table")
table_source = config.catalog("cat").schema("test").table("table")


def test_databricks_filter_modification() -> None:
    job = sql_source.all_columns()

    filtered_job = job.filter(pl.col("a") == 10)

    assert isinstance(filtered_job, UCSqlJob)
    filter = filtered_job.filter_exp
    assert filter


def test_databricks_sql_depends_on() -> None:
    locs = sql_source.depends_on()

    assert locs == {FeatureLocation.feature_view("some_table")}


def test_databricks_table_modification() -> None:
    from pyspark.sql.functions import col

    limit_size = 100
    job = table_source.all_columns(limit_size)

    assert isinstance(job, UnityCatalogTableAllJob)
    assert job._limit == limit_size

    filter_exp = pl.col("a") == 10
    filter_job = job.filter(filter_exp)

    assert isinstance(filter_job, UnityCatalogTableAllJob)
    assert filter_job.where is not None
    assert filter_exp.meta.eq(filter_job.where.to_polars())  # type: ignore
    assert str(filter_job.where.to_spark()) == str(col("a") == 10)

    another_exp = pl.col("b") == "test"
    another_job = filter_job.filter(another_exp)

    assert isinstance(another_job, UnityCatalogTableAllJob)
    assert another_job.where is not None
    assert (filter_exp & another_exp).meta.eq(another_job.where.to_polars())  # type: ignore
    assert str(another_job.where.to_spark()) == str(
        (col("a") == 10) & (col("b") == "test")
    )
