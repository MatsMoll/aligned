from datetime import timedelta

import pandas as pd
import pytest
from freezegun import freeze_time

from aligned.enricher import TimespanSelector
from aligned.sources.local import CsvFileSource


@pytest.mark.asyncio
async def test_statistic_enricher(scan_with_datetime: CsvFileSource) -> None:
    columns = {'fractal_dimension_worst', 'symmetry_worst'}
    file = await scan_with_datetime.mean(columns=columns).as_df()
    limit_file = await scan_with_datetime.mean(columns=columns, limit=3).as_df()

    assert set(file.index) == columns
    expected_values = {'symmetry_worst': 0.3654, 'fractal_dimension_worst': 0.0985}
    expected_series = pd.Series([expected_values[col] for col in limit_file.index], index=limit_file.index)
    pd.testing.assert_series_equal(limit_file, expected_series, atol=0.001)


@pytest.mark.asyncio
async def test_statistic_enricher_with_limit(scan_with_datetime: CsvFileSource) -> None:
    columns = {'fractal_dimension_worst', 'symmetry_worst'}
    limit_file = await scan_with_datetime.mean(columns=columns, limit=3).as_df()

    assert set(limit_file.index) == columns
    expected_values = {'symmetry_worst': 0.3654, 'fractal_dimension_worst': 0.0985}
    expected_series = pd.Series([expected_values[col] for col in limit_file.index], index=limit_file.index)
    pd.testing.assert_series_equal(limit_file, expected_series, atol=0.001)


@freeze_time('2020-01-11')
@pytest.mark.asyncio
async def test_statistic_enricher_with_timespand(scan_with_datetime: CsvFileSource) -> None:

    columns = {'fractal_dimension_worst', 'symmetry_worst'}
    limit_file = await scan_with_datetime.mean(
        columns=columns, time=TimespanSelector(timespand=timedelta(days=3), time_column='created_at')
    ).as_df()

    assert set(limit_file.index) == columns
    expected_values = {'symmetry_worst': 0.398, 'fractal_dimension_worst': 0.14326}
    expected_series = pd.Series([expected_values[col] for col in limit_file.index], index=limit_file.index)
    pd.testing.assert_series_equal(limit_file, expected_series, atol=0.001)
