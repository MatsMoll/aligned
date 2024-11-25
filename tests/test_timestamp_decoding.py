from datetime import datetime, timezone, timedelta
import polars as pl
from aligned import feature_view, FileSource, String, Timestamp, EventTimestamp, Int32
import pytest
from aligned.data_file import DataFileReference

from aligned.schemas.date_formatter import DateFormatter


@feature_view(
    name='timestamp',
    source=FileSource.parquet_at('test_data/data/placeholder.parquet'),
)
class TimestampView:
    id = Int32().as_entity()
    other = String()
    et = EventTimestamp()
    timestamp = Timestamp()


data = pl.DataFrame(
    {
        'id': [1, 2, 3],
        'other': ['foo', 'bar', 'baz'],
        'et': [
            datetime.now(timezone.utc),
            datetime.now(timezone.utc) - timedelta(days=1),
            datetime.now(timezone.utc) - timedelta(days=2),
        ],
        'timestamp': [
            datetime.now(timezone.utc),
            datetime.now(timezone.utc) + timedelta(days=1),
            datetime.now(timezone.utc) + timedelta(days=2),
        ],
    }
)


@pytest.mark.asyncio
async def test_unix_timestamp() -> None:

    formatter = DateFormatter.unix_timestamp()

    sources_to_test: list[DataFileReference] = [
        FileSource.parquet_at('test_data/data/parquet_unix.parquet', date_formatter=formatter),
        FileSource.csv_at('test_data/data/csv_unix.csv', date_formatter=formatter),
        # FileSource.delta_at("test_data/data/delta_unix", date_formatter=formatter),
    ]

    converted_data = data.with_columns([formatter.encode_polars('et'), formatter.encode_polars('timestamp')])

    for source in sources_to_test:
        await source.write_polars(converted_data.lazy())

        TimestampView.metadata.source = source  # type: ignore

        view = TimestampView.query()  # type: ignore
        assert view.view.source == source

        df = await view.all().to_polars()

        assert df.select(data.columns).equals(data)

        df = await view.features_for(
            {
                'id': [1, 2],
            }
        ).to_polars()

        assert df.select(data.columns).equals(data.filter(pl.col('id').is_in([1, 2])))


@pytest.mark.asyncio
async def test_iso_timestamp() -> None:

    formatter = DateFormatter.iso_8601()

    sources_to_test = [
        FileSource.csv_at('test_data/data/csv_iso.csv', date_formatter=formatter),
        FileSource.parquet_at('test_data/data/parquet_iso.parquet', date_formatter=formatter),
        # FileSource.delta_at("test_data/data/delta", date_formatter=formatter),
    ]

    converted_data = data.with_columns([formatter.encode_polars('et'), formatter.encode_polars('timestamp')])

    for source in sources_to_test:
        await source.write_polars(converted_data.lazy())  # type: ignore

        TimestampView.metadata.source = source  # type: ignore

        view = TimestampView.query()  # type: ignore
        assert view.view.source == source

        df = await view.all().to_polars()

        assert df.select(data.columns).equals(data)

        df = await view.features_for(
            {
                'id': [1, 2],
            }
        ).to_polars()

        assert df.select(data.columns).equals(data.filter(pl.col('id').is_in([1, 2])))
