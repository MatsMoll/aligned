import pytest

from aligned import feature_view, EventTimestamp, Int32, Timestamp
from aligned.sources.local import CsvFileSource


@pytest.mark.asyncio
async def test_datetime_timezone_conversion(scan_with_datetime: CsvFileSource) -> None:
    @feature_view(
        name='test_feature',
        source=scan_with_datetime,
    )
    class TestFeatureUtc:

        scan_id = Int32().as_entity()
        created_at = EventTimestamp(time_zone='UTC')

    @feature_view(
        name='test_feature',
        source=scan_with_datetime,
    )
    class TestFeatureUtcTimestamp:

        scan_id = Int32().as_entity()
        created_at = Timestamp(time_zone='UTC')

    @feature_view(
        name='test_feature',
        source=scan_with_datetime,
    )
    class TestFeatureNone:

        scan_id = Int32().as_entity()
        created_at = EventTimestamp(time_zone=None)

    @feature_view(
        name='test_feature',
        source=scan_with_datetime,
    )
    class TestFeatureNoneTimestamp:

        scan_id = Int32().as_entity()
        created_at = Timestamp(time_zone=None)

    data_utc = await TestFeatureUtc.query().all().to_polars()
    data_none = await TestFeatureNone.query().all().to_polars()
    data_utc_timestamp = await TestFeatureUtcTimestamp.query().all().to_polars()
    data_none_timestamp = await TestFeatureNoneTimestamp.query().all().to_polars()

    assert data_utc['created_at'].dtype.time_zone == 'UTC'
    assert data_none['created_at'].dtype.time_zone is None
    assert data_utc_timestamp['created_at'].dtype.time_zone == 'UTC'
    assert data_none_timestamp['created_at'].dtype.time_zone is None
