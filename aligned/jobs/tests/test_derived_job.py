from datetime import datetime, timedelta

import pandas as pd
import pytest

from aligned import feature_view, Float, String, FileSource
from aligned.compiler.model import model_contract
from aligned.feature_store import FeatureStore
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


@feature_view(name='transactions', source=FileSource.csv_at('test_data/transactions.csv'))
class Transaction:

    transaction_id = String().as_entity()

    user_id = String().fill_na('some_user_id')

    amount = Float()
    abs_amount = abs(amount)

    is_expence = amount < 0
    is_income = amount > 0


Expences = Transaction.filter(name='expence', where=lambda view: view.is_expence)  # type: ignore
Income = Transaction.filter(name='income', where=lambda view: view.is_income)  # type: ignore

expences = Expences()


@feature_view(name='expence_agg', source=Expences)
class ExpenceAgg:
    user_id = String().as_entity()

    amount_agg = expences.abs_amount.aggregate()

    total_amount = amount_agg.sum()


IncomeAgg = ExpenceAgg.with_source(named='income_agg', source=Income)  # type: ignore

income_agg = IncomeAgg()


@model_contract(
    name='model',
    features=[
        expences.abs_amount,
        expences.is_expence,
        income_agg.total_amount,
    ],
)
class Model:
    user_id = String().as_entity()

    pred_amount = expences.amount.as_regression_label()


def feature_store() -> FeatureStore:
    store = FeatureStore.experimental()

    views = [Transaction, Expences, Income, ExpenceAgg, IncomeAgg]
    for view in views:
        store.add_compiled_view(view.compile())

    store.add_compiled_model(Model.compile())

    return store


@pytest.mark.asyncio
async def test_aggregate_over_derived() -> None:

    data = await IncomeAgg.query().all().to_lazy_polars()

    df = data.collect()

    assert df.height == 2


@pytest.mark.asyncio
async def test_aggregate_over_derived_fact() -> None:

    store = feature_store()

    data = await store.features_for(
        entities={'user_id': ['a', 'b']}, features=['income_agg:total_amount']
    ).to_lazy_polars()

    df = data.collect()

    assert df.height == 2


@pytest.mark.asyncio
async def test_model_with_label_multiple_views() -> None:

    store = feature_store()

    entities = await store.feature_view('expence').all().to_pandas()

    data_job = store.model('model').with_labels().features_for(entities)
    data = await data_job.to_pandas()

    expected_df = pd.DataFrame(
        {
            'transaction_id': ['b', 'd', 'q', 'e'],
            'user_id': ['b', 'b', 'a', 'a'],
            'total_amount': [109.0, 109.0, 120.0, 120.0],
            'is_expence': [True, True, True, True],
            'abs_amount': [20, 100, 20, 100],
            'amount': [-20.0, -100.0, -20.0, -100.0],
        }
    )

    assert data.labels.shape[0] != 0
    assert data.input.shape[1] == 3
    assert data.input.shape[0] != 0

    assert data.data.sort_values(['user_id', 'transaction_id'])[expected_df.columns].equals(
        expected_df.sort_values(['user_id', 'transaction_id'])
    )
