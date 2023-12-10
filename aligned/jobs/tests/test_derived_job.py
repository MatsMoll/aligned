from datetime import datetime, timedelta

import pandas as pd
import pytest

from aligned import feature_view, Float, String, FileSource
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


@pytest.mark.asyncio
async def test_aggregate_over_derived() -> None:

    data = await IncomeAgg.query().all().to_polars()

    df = data.collect()

    assert df.height == 2
