import pytest
from aligned.compiler.feature_factory import EventTimestamp, Int32, String

from aligned.feature_store import FeatureStore
from aligned.feature_view.feature_view import feature_view
from aligned.schemas.transformation import SupportedTransformations
from aligned.sources.local import FileSource, CsvFileSource


@pytest.mark.asyncio
async def test_transformations() -> None:
    supported = SupportedTransformations.shared()

    for transformation in supported.types.values():
        await transformation.run_transformation_test_pandas()


@pytest.mark.asyncio
async def test_polars_transformation() -> None:
    supported = SupportedTransformations.shared()

    for transformation in supported.types.values():
        await transformation.run_transformation_test_polars()


@pytest.mark.asyncio
async def test_transformations_in_feture_view(alot_of_transforation_feature_store: FeatureStore) -> None:
    store = alot_of_transforation_feature_store

    amount = 100

    data = await store.feature_view('titanic').all(limit=amount).to_pandas()

    assert data.shape[0] == amount


@pytest.mark.asyncio
async def test_aggregations_on_all() -> None:
    @feature_view(name='test_agg', source=FileSource.parquet_at('test_data/credit_history.parquet'))
    class TestAgg:
        dob_ssn = String().as_entity()

        event_timestamp = EventTimestamp()

        credit_card_due = Int32()
        student_loan_due = Int32()

        credit_card_due_sum = credit_card_due.aggregate().over(days=1).sum()
        student_loan_due_mean = student_loan_due.aggregate().over(days=1).mean()

    df = await TestAgg.query().all().to_pandas()  # type: ignore
    assert df.shape[0] == 6


@pytest.mark.asyncio
async def test_aggregations_on_all_no_window() -> None:
    @feature_view(name='test_agg', source=FileSource.parquet_at('test_data/credit_history.parquet'))
    class TestAgg:
        dob_ssn = String().as_entity()

        event_timestamp = EventTimestamp()

        credit_card_due = Int32()
        student_loan_due = Int32()

        credit_card_due_sum = credit_card_due.aggregate().sum()
        student_loan_due_mean = student_loan_due.aggregate().mean()

    df = await TestAgg.query().all().to_pandas()  # type: ignore
    assert df.shape[0] == 3


@pytest.mark.asyncio
async def test_aggregations_on_all_no_window_materialised() -> None:
    materialized_source = FileSource.parquet_at('test_data/credit_history_mater.parquet')

    @feature_view(
        name='test_agg',
        source=FileSource.parquet_at('test_data/credit_history.parquet'),
        materialized_source=materialized_source,
    )
    class TestAgg:
        dob_ssn = String().as_entity()

        event_timestamp = EventTimestamp()

        credit_card_due = Int32()
        student_loan_due = Int32()

        credit_card_due_sum = credit_card_due.aggregate().sum()
        student_loan_due_mean = student_loan_due.aggregate().mean()

    org_values_job = TestAgg.query().using_source(TestAgg.metadata.source).all()  # type: ignore
    await org_values_job.write_to_source(materialized_source)

    values = await org_values_job.to_polars()
    df = await TestAgg.query().all().to_polars()  # type: ignore

    assert df.sort('dob_ssn').collect().frame_equal(values.sort('dob_ssn').select(df.columns).collect())


@pytest.mark.asyncio
async def test_transform_entity(titanic_source: CsvFileSource) -> None:
    @feature_view(name='titanic', source=titanic_source)
    class Titanic:

        passenger_id = Int32()
        cabin = String().fill_na(passenger_id).as_entity()

        sex = String()

    data = await Titanic.query().all().to_pandas()  # type: ignore

    assert data['cabin'].isnull().sum() == 0
