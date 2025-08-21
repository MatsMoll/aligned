import polars as pl
from sqlglot import exp
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
    filter = filtered_job.query.find(exp.Where)
    assert filter

    assert filter.sql(dialect="spark") == "WHERE (a = 10)"


def test_databricks_sql_depends_on() -> None:
    locs = sql_source.depends_on()

    assert locs == {FeatureLocation.feature_view("some_table")}


def test_databricks_table_modification() -> None:
    limit_size = 100
    job = table_source.all_columns(limit_size)

    assert isinstance(job, UnityCatalogTableAllJob)
    assert job._limit == limit_size

    filter_job = job.filter(pl.col("a") == 10)

    assert isinstance(filter_job, UnityCatalogTableAllJob)
    assert filter_job.where == "(a = 10)"

    another_job = filter_job.filter(pl.col("b") == "test")

    assert isinstance(another_job, UnityCatalogTableAllJob)
    assert filter_job.where == "(a = 10) AND (b = 'test')"
