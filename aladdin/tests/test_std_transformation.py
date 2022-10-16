from datetime import timedelta

import pytest
from freezegun import freeze_time

from aladdin import Entity, EventTimestamp, FeatureView, FeatureViewMetadata, Float, Int32
from aladdin.exceptions import InvalidStandardScalerArtefact
from aladdin.local.source import CsvFileSource


@pytest.mark.asyncio
async def test_standard_scaler_should_fail(scan_with_datetime: CsvFileSource) -> FeatureView:
    class BreastDiagnoseFeatureView(FeatureView):

        metadata = FeatureViewMetadata(
            name='breast_features',
            description='Features defining a scan and diagnose of potential cancer cells',
            tags={},
            batch_source=scan_with_datetime,
        )

        scan_id = Entity(dtype=Int32())

        created_at = EventTimestamp()

        fractal_dimension_worst = Float()
        scaled_fraction = fractal_dimension_worst.standard_scaled(timespan=timedelta(days=10))

    with pytest.raises(InvalidStandardScalerArtefact) as error:
        await BreastDiagnoseFeatureView.compile()
    assert 'fractal_dimension_worst' in str(error.value)


@freeze_time('2020-01-11')
def test_standard_scaler_should_not_fail(scan_with_datetime: CsvFileSource) -> FeatureView:
    class BreastDiagnoseFeatureView(FeatureView):

        metadata = FeatureViewMetadata(
            name='breast_features',
            description='Features defining a scan and diagnose of potential cancer cells',
            tags={},
            batch_source=scan_with_datetime,
        )

        scan_id = Entity(dtype=Int32())

        created_at = EventTimestamp()

        fractal_dimension_worst = Float()
        scaled_fraction = fractal_dimension_worst.standard_scaled(timespan=timedelta(days=10))

    assert BreastDiagnoseFeatureView.compile()
