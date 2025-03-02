from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from aligned.lazy_imports import pandas as pd
from aligned import feature_view, Float32, String, FileSource
from aligned.compiler.model import model_contract
from aligned.feature_store import ContractStore
from aligned.local.job import FileFullJob
from aligned.retrieval_job import DerivedFeatureJob, RetrievalRequest
from aligned.sources.local import LiteralReference


@pytest.mark.asyncio
async def test_derived_pandas(retrieval_request_with_derived: RetrievalRequest) -> None:
    date = datetime(year=2022, month=1, day=1)
    one_day = timedelta(days=1)
    job = DerivedFeatureJob(
        job=FileFullJob(
            LiteralReference(
                pd.DataFrame(
                    {
                        "id": [1, 2, 3, 4, 5],
                        "c": [3, 4, 2, 3, 4],
                        "d": [1, 1, 1, 2, 4],
                        "created_at": [
                            date,
                            date,
                            date + one_day,
                            date + one_day,
                            date + one_day,
                        ],
                    }
                )
            ),
            request=retrieval_request_with_derived,
        ),
        requests=[retrieval_request_with_derived],
    )

    data = await job.to_pandas()

    assert set(data.columns) == {"id", "c", "d", "created_at", "c+d"}
    assert data.shape[0] == 5


@pytest.mark.asyncio
async def test_derived_polars(retrieval_request_with_derived: RetrievalRequest) -> None:
    date = datetime(year=2022, month=1, day=1)
    one_day = timedelta(days=1)
    job = DerivedFeatureJob(
        job=FileFullJob(
            LiteralReference(
                pd.DataFrame(
                    {
                        "id": [1, 2, 3, 4, 5],
                        "c": [3, 4, 2, 3, 4],
                        "d": [1, 1, 1, 2, 4],
                        "created_at": [
                            date,
                            date,
                            date + one_day,
                            date + one_day,
                            date + one_day,
                        ],
                    }
                )
            ),
            request=retrieval_request_with_derived,
        ),
        requests=[retrieval_request_with_derived],
    )

    data = await job.to_pandas()

    assert set(data.columns) == {"id", "c", "d", "created_at", "c+d"}
    assert data.shape[0] == 5


@feature_view(
    name="transactions", source=FileSource.csv_at("test_data/transactions.csv")
)
class Transaction:
    transaction_id = String().as_entity()

    user_id = String().fill_na("some_user_id")

    amount = Float32()
    abs_amount = abs(amount)

    is_expense = amount < 0
    is_income = amount > 0


Expenses = Transaction.filter(name="expense", where=lambda view: view.is_expense)  # type: ignore
Income = Transaction.filter(name="income", where=lambda view: view.is_income)  # type: ignore

expenses = Expenses()


@feature_view(name="expense_agg", source=Expenses)
class ExpenseAgg:
    user_id = String().as_entity()

    amount_agg = expenses.abs_amount.aggregate()

    total_amount = amount_agg.sum()


IncomeAgg = ExpenseAgg.with_source(named="income_agg", source=Income)  # type: ignore

income_agg = IncomeAgg()


@model_contract(
    name="model",
    input_features=[
        expenses.abs_amount,
        expenses.is_expense,
        income_agg.total_amount,
    ],
)
class Model:
    user_id = String().as_entity()

    pred_amount = expenses.amount.as_regression_label()


def feature_store() -> ContractStore:
    store = ContractStore.experimental()

    views = [Transaction, Expenses, Income, ExpenseAgg, IncomeAgg]
    for view in views:
        store.add_compiled_view(view.compile())

    store.add_compiled_model(Model.compile())

    return store


@pytest.mark.asyncio
async def test_without_derived_features():
    df = await Transaction.query().all().to_polars()

    assert "is_expense" in df.columns

    without_job = Transaction.query().all().remove_derived_features()
    without_df = await without_job.to_polars()

    assert "is_expense" not in without_df.columns

    feature_columns = without_job.request_result.feature_columns
    assert "is_expense" not in feature_columns


def test_with_schema() -> None:
    Test = Transaction.with_schema(
        name="test",
        source=FileSource.parquet_at("test_data/transactions.parquet"),
        entities=dict(  # noqa: C408
            other_id=String(),
        ),
        additional_features=dict(  # noqa: C408
            other=Float32(),
        ),
    )
    transaction = Transaction.compile()

    assert len(transaction.derived_features) > 1

    view = Test.compile()
    assert len(view.entities) == 1

    assert len(view.derived_features) == 0
    assert len(view.aggregated_features) == 0
    assert (
        len(view.features)
        == len({feat.name for feat in transaction.full_schema - transaction.entities})
        + 1
    )

    assert list(view.entities)[0].name == "other_id"


@pytest.mark.asyncio
async def test_polars_filter_source() -> None:
    Expenses = Transaction.filter(name="expense", where=pl.col("amount") > 0)  # type: ignore
    data = await Expenses.query().all().to_lazy_polars()

    df = data.collect()

    assert df.height == 4


@pytest.mark.asyncio
async def test_aggregate_over_derived() -> None:
    data = await IncomeAgg.query().all().to_lazy_polars()

    df = data.collect()

    assert df.height == 2


@pytest.mark.asyncio
async def test_aggregate_over_derived_fact() -> None:
    store = feature_store()

    data = await store.features_for(
        entities={"user_id": ["a", "b"]}, features=["income_agg:total_amount"]
    ).to_lazy_polars()

    df = data.collect()

    assert df.height == 2


@pytest.mark.asyncio
async def test_model_with_label_multiple_views() -> None:
    store = feature_store()
    entities = await (
        store.feature_view("expense")
        .select_columns(["transaction_id", "user_id"])
        .to_pandas()
    )

    data_job = store.model("model").with_labels().features_for(entities)
    data = await data_job.to_pandas()

    expected_df = pd.DataFrame(
        {
            "transaction_id": ["b", "d", "q", "e"],
            "user_id": ["b", "b", "a", "a"],
            "total_amount": [109.0, 109.0, 120.0, 120.0],
            "is_expense": [True, True, True, True],
            "abs_amount": [20.0, 100.0, 20.0, 100.0],
            "amount": [-20.0, -100.0, -20.0, -100.0],
        }
    )
    expected_df["total_amount"] = expected_df["total_amount"].astype("float32")
    expected_df["abs_amount"] = expected_df["abs_amount"].astype("float32")
    expected_df["amount"] = expected_df["amount"].astype("float32")

    assert data.labels.shape[0] != 0
    assert data.input.shape[1] == 3
    assert data.input.shape[0] != 0

    result = data.data.sort_values(["user_id", "transaction_id"])[expected_df.columns]
    assert result.equals(expected_df.sort_values(["user_id", "transaction_id"]))
