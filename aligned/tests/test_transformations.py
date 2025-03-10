import pytest
from aligned.compiler.feature_factory import EventTimestamp, Int32, String, Float32, List

from aligned.feature_store import ContractStore
from aligned.feature_view.feature_view import feature_view
from aligned.schemas.transformation import SupportedTransformations
from aligned.sources.in_mem_source import InMemorySource
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
async def test_transformations_in_feture_view(alot_of_transforation_feature_store: ContractStore) -> None:
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
        student_loan_due_mean = student_loan_due.aggregate().over(days=1).mean().with_tag('mean')

    df = await TestAgg.query().all().to_pandas()  # type: ignore
    assert df.shape[0] == 6

    all_features = TestAgg.compile().request_all.request_result.features
    assert len(all_features) == 4

    with_tag = [feature for feature in all_features if feature.tags]
    assert len(with_tag) == 1


@pytest.mark.asyncio
async def test_aggregations_on_all_no_window() -> None:
    import polars as pl

    @feature_view(name='test_agg', source=FileSource.parquet_at('test_data/credit_history.parquet'))
    class TestAgg:
        dob_ssn = String().as_entity()

        event_timestamp = EventTimestamp()

        credit_card_due = Int32()
        student_loan_due = Int32()

        credit_card_due_sum = credit_card_due.aggregate().sum()
        student_loan_due_mean = student_loan_due.aggregate().mean()

        custom_mean_aggregation = student_loan_due.polars_aggregation(
            pl.col('student_loan_due').mean(),
            as_type=Float32(),
        )
        custom_mean_aggregation_using_features = Float32().polars_aggregation_using_features(
            using_features=[student_loan_due],
            aggregation=pl.col('student_loan_due').mean(),
        )
        custom_sum_aggregation = credit_card_due.polars_aggregation(
            pl.col('credit_card_due').sum(),
            as_type=Float32(),
        )

    df = await TestAgg.query().all().to_pandas()  # type: ignore
    assert df.shape[0] == 3

    assert df['custom_mean_aggregation'].equals(df['student_loan_due_mean'])
    assert df['custom_mean_aggregation'].equals(df['custom_mean_aggregation_using_features'])
    assert df['custom_sum_aggregation'].equals(df['credit_card_due_sum'])


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

    values = await org_values_job.to_lazy_polars()
    descrete_values = await org_values_job.to_polars()
    df = await TestAgg.query().all().to_lazy_polars()  # type: ignore

    assert df.sort('dob_ssn').collect().equals(values.sort('dob_ssn').select(df.columns).collect())
    assert descrete_values.sort('dob_ssn').equals(
        values.sort('dob_ssn').select(descrete_values.columns).collect()
    )


@pytest.mark.asyncio
async def test_transform_entity(titanic_source: CsvFileSource) -> None:
    @feature_view(name='titanic', source=titanic_source)
    class Titanic:

        passenger_id = Int32()
        cabin = String().fill_na(passenger_id).as_entity()

        sex = String()

    data = await Titanic.query().all().to_pandas()  # type: ignore

    assert data['cabin'].isnull().sum() == 0


@pytest.mark.asyncio
async def test_fill_optional_column_bug(titanic_source: CsvFileSource) -> None:
    @feature_view(name='test_fill', source=titanic_source)
    class TestFill:

        passenger_id = String().as_entity()

        cabin = String().is_optional()
        some_new_column = Int32().is_optional().fill_na(0)
        some_string = String().is_optional().fill_na('some_string')

        is_male = cabin == 'male'

    df = await TestFill.query().all().to_polars()

    assert df['some_new_column'].is_null().sum() == 0
    assert df['some_string'].is_null().sum() == 0


@pytest.mark.asyncio
async def test_load_features() -> None:
    import polars as pl

    @feature_view(source=InMemorySource.from_values({'passenger_id': [1, 2, 3], 'age': [24, 20, 30]}))
    class Test:
        passenger_id = Int32().as_entity()
        age = Int32()

    @feature_view(source=InMemorySource.empty())
    class Other:
        some_value = Int32()

        lookup_id = some_value.transform_polars(pl.lit([2, 1]), as_dtype=List(Int32()))
        age_value = Test().age.for_entities({'passenger_id': lookup_id})

    store = ContractStore.empty()
    store.add(Test)
    store.add(Other)

    df = await store.feature_view(Other).features_for({'some_value': [1, 10, 5]}).to_polars()

    assert Other().age_value._loads_feature is not None
    assert df['age_value'].null_count() == 0
