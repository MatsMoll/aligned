from dataclasses import dataclass
from math import ceil, floor

import polars as pl
import pytest
import pytest_asyncio

from aligned import (
    Bool,
    Entity,
    EventTimestamp,
    FeatureView,
    FeatureViewMetadata,
    FileSource,
    Float,
    Int32,
    Int64,
    ModelContract,
    RedisConfig,
    String,
    TextVectoriserModel,
)
from aligned.feature_store import FeatureStore
from aligned.feature_view.combined_view import CombinedFeatureView, CombinedFeatureViewMetadata
from aligned.retrival_job import DerivedFeatureJob, RetrivalJob, RetrivalRequest
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReferance, FeatureType
from aligned.schemas.record_coders import JsonRecordCoder
from aligned.sources.local import CsvFileSource, FileFullJob, LiteralReference, ParquetFileSource


@pytest.fixture
def retrival_request_without_derived() -> RetrivalRequest:
    return RetrivalRequest(
        name='test',
        location=FeatureLocation.feature_view('test'),
        entities={Feature(name='id', dtype=FeatureType('').int32)},
        features={
            Feature(name='a', dtype=FeatureType('').int32),
            Feature(name='b', dtype=FeatureType('').int32),
        },
        derived_features=set(),
        event_timestamp=None,
    )


@pytest.fixture
def retrival_job(retrival_request_without_derived: RetrivalRequest) -> RetrivalJob:
    import pandas as pd

    return FileFullJob(
        LiteralReference(pd.DataFrame({'id': [1, 2, 3, 4, 5], 'a': [3, 4, 2, 3, 4], 'b': [1, 1, 1, 2, 4]})),
        request=retrival_request_without_derived,
    )


@pytest.fixture
def retrival_request_with_derived() -> RetrivalRequest:
    from aligned.schemas.feature import EventTimestamp as TimestampFeature
    from aligned.schemas.transformation import Addition

    return RetrivalRequest(
        name='test_with_ts',
        location=FeatureLocation.feature_view('test_with_ts'),
        entities={Feature(name='id', dtype=FeatureType('').int32)},
        features={
            Feature(name='c', dtype=FeatureType('').int32),
            Feature(name='d', dtype=FeatureType('').int32),
        },
        derived_features={
            DerivedFeature(
                name='c+d',
                dtype=FeatureType('').int32,
                depending_on={
                    FeatureReferance(
                        name='c',
                        location=FeatureLocation.feature_view('test_with_ts'),
                        dtype=FeatureType('').int32,
                    ),
                    FeatureReferance(
                        name='d',
                        location=FeatureLocation.feature_view('test_with_ts'),
                        dtype=FeatureType('').int32,
                    ),
                },
                transformation=Addition(front='c', behind='d'),
                depth=1,
            )
        },
        event_timestamp=TimestampFeature(name='created_at'),
    )


@pytest.fixture
def retrival_job_with_timestamp(retrival_request_with_derived: RetrivalRequest) -> RetrivalJob:
    from datetime import datetime, timedelta

    import pandas as pd

    date = datetime(year=2022, month=1, day=1)
    one_day = timedelta(days=1)
    return DerivedFeatureJob(
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


@pytest.fixture
def combined_retrival_request() -> RetrivalRequest:
    from aligned.schemas.transformation import Addition

    return RetrivalRequest(
        name='combined',
        location=FeatureLocation.combined_view('combined'),
        entities={Feature(name='id', dtype=FeatureType('').int32)},
        features=set(),
        derived_features={
            DerivedFeature(
                name='a+c+d',
                dtype=FeatureType('').int32,
                depending_on={
                    FeatureReferance(
                        name='c+d',
                        location=FeatureLocation.feature_view('test_with_ts'),
                        dtype=FeatureType('').int32,
                    ),
                    FeatureReferance(
                        name='a', location=FeatureLocation.feature_view('test'), dtype=FeatureType('').int32
                    ),
                },
                transformation=Addition(front='c+d', behind='a'),
                depth=2,
            )
        },
    )


@pytest.fixture
def scan_without_datetime() -> CsvFileSource:
    return FileSource.csv_at(path='test_data/data.csv', mapping_keys={'id': 'scan_id'})


@pytest.fixture
def breast_scan_feature_viewout_with_datetime(scan_without_datetime: CsvFileSource) -> FeatureView:
    class BreastDiagnoseFeatureView(FeatureView):

        metadata = FeatureViewMetadata(
            name='breast_features',
            description='Features defining a scan and diagnose of potential cancer cells',
            tags={},
            batch_source=scan_without_datetime,
        )

        scan_id = Entity(dtype=Int32())
        diagnosis = String().description('The given diagnose. M for malignant, and B for benigne')
        is_malignant = (diagnosis == 'M').description('If the scanned cells was diagnosed as dangerous')

        radius_mean = Float()
        radius_se = Float()
        radius_worst = Float()

        texture_mean = Float()
        texture_se = Float()
        texture_worst = Float()

        perimeter_mean = Float()
        perimeter_se = Float()
        perimeter_worst = Float()

        area_mean = Float()
        area_se = Float()
        area_worst = Float()

        smoothness_mean = Float()
        smoothness_se = Float()
        smoothness_worst = Float()

        compactness_mean = Float()
        compactness_se = Float()
        compactness_worst = Float()

        concavity_mean = Float()
        concavity_se = Float()
        concavity_worst = Float()

        concave_points_mean = Float()
        concave_points_se = Float()
        concave_points_worst = Float()

        symmetry_mean = Float()
        symmetry_se = Float()
        symmetry_worst = Float()

        fractal_dimension_mean = Float()
        fractal_dimension_se = Float()
        fractal_dimension_worst = Float()

    return BreastDiagnoseFeatureView()


@pytest_asyncio.fixture
async def breast_scan_without_timestamp_feature_store(
    breast_scan_feature_viewout_with_datetime: FeatureView,
) -> FeatureStore:
    store = FeatureStore.experimental()
    store.add_feature_view(breast_scan_feature_viewout_with_datetime)
    return store


@pytest.fixture
def scan_with_datetime() -> CsvFileSource:
    return FileSource.csv_at(path='test_data/data-with-datetime.csv')


@pytest.fixture
def breast_scan_feature_view_with_datetime(scan_with_datetime: CsvFileSource) -> FeatureView:
    class BreastDiagnoseFeatureView(FeatureView):

        metadata = FeatureViewMetadata(
            name='breast_features',
            description='Features defining a scan and diagnose of potential cancer cells',
            tags={},
            batch_source=scan_with_datetime,
        )

        scan_id = Entity(dtype=Int32())

        created_at = EventTimestamp()

        diagnosis = String().description('The given diagnose. M for malignant, and B for benigne')
        is_malignant = (diagnosis == 'M').description('If the scanned cells was diagnosed as dangerous')

        radius_mean = Float()
        radius_se = Float()
        radius_worst = Float()

        texture_mean = Float()
        texture_se = Float()
        texture_worst = Float()

        perimeter_mean = Float()
        perimeter_se = Float()
        perimeter_worst = Float()

        area_mean = Float()
        area_se = Float()
        area_worst = Float()

        smoothness_mean = Float()
        smoothness_se = Float()
        smoothness_worst = Float()

        compactness_mean = Float()
        compactness_se = Float()
        compactness_worst = Float()

        concavity_mean = Float()
        concavity_se = Float()
        concavity_worst = Float()

        concave_points_mean = Float()
        concave_points_se = Float()
        concave_points_worst = Float()

        symmetry_mean = Float()
        symmetry_se = Float()
        symmetry_worst = Float()

        fractal_dimension_mean = Float()
        fractal_dimension_se = Float()
        fractal_dimension_worst = Float()

    return BreastDiagnoseFeatureView()


@pytest.fixture
def breast_scan_feature_view_with_datetime_and_aggregation(scan_with_datetime: CsvFileSource) -> FeatureView:
    class BreastDiagnoseFeatureView(FeatureView):

        metadata = FeatureViewMetadata(
            name='breast_features',
            description='Features defining a scan and diagnose of potential cancer cells',
            tags={},
            batch_source=scan_with_datetime,
        )

        scan_id = Entity(dtype=Int32())

        created_at = EventTimestamp()

        diagnosis = String().description('The given diagnose. M for malignant, and B for benigne')
        is_malignant = (diagnosis == 'M').description('If the scanned cells was diagnosed as dangerous')

        radius_mean = Float()
        radius_se = Float()
        radius_worst = Float()

        texture_mean = Float()
        texture_se = Float()
        texture_worst = Float()

        perimeter_mean = Float()
        perimeter_se = Float()
        perimeter_worst = Float()

        area_mean = Float()
        area_se = Float()
        area_worst = Float()

        smoothness_mean = Float()
        smoothness_se = Float()
        smoothness_worst = Float()

        compactness_mean = Float()
        compactness_se = Float()
        compactness_worst = Float()

        concavity_mean = Float()
        concavity_se = Float()
        concavity_worst = Float()

        concave_points_mean = Float()
        concave_points_se = Float()
        concave_points_worst = Float()

        symmetry_mean = Float()
        symmetry_se = Float()
        symmetry_worst = Float()

        fractal_dimension_mean = Float()
        fractal_dimension_se = Float()
        fractal_dimension_worst = Float()

    return BreastDiagnoseFeatureView()


@pytest_asyncio.fixture
async def breast_scan_with_timestamp_feature_store(
    breast_scan_feature_view_with_datetime: FeatureView,
) -> FeatureStore:
    store = FeatureStore.experimental()
    store.add_feature_view(breast_scan_feature_view_with_datetime)
    return store


@pytest_asyncio.fixture
async def breast_scan_with_timestamp_and_aggregation_feature_store(
    breast_scan_feature_view_with_datetime_and_aggregation: FeatureView,
) -> FeatureStore:
    store = FeatureStore.experimental()
    store.add_feature_view(breast_scan_feature_view_with_datetime_and_aggregation)
    return store


@pytest.fixture
def titanic_source() -> CsvFileSource:
    return FileSource.csv_at(
        'test_data/titanic_dataset.csv',
        mapping_keys={
            'PassengerId': 'passenger_id',
            'Age': 'age',
            'Name': 'name',
            'Sex': 'sex',
            'Survived': 'survived',
            'SibSp': 'sibsp',
            'Cabin': 'cabin',
        },
    )


@pytest.fixture
def titanic_source_scd() -> CsvFileSource:
    return FileSource.csv_at(
        'test_data/titanic_scd_data.csv',
        mapping_keys={
            'PassengerId': 'passenger_id',
            'Age': 'age',
            'Sex': 'sex',
            'Survived': 'survived',
            'SibSp': 'sibsp',
            'UpdatedAt': 'updated_at',
        },
    )


@pytest.fixture
def titanic_source_parquet() -> CsvFileSource:
    return FileSource.parquet_at('test_data/titanic.parquet')


@pytest.fixture
def titanic_feature_view(titanic_source: CsvFileSource) -> FeatureView:
    class TitanicPassenger(FeatureView):

        metadata = FeatureViewMetadata(
            name='titanic', description='Some features from the titanic dataset', batch_source=titanic_source
        )

        passenger_id = Entity(dtype=Int32())

        # Input values
        age = (
            Float().is_required().lower_bound(0).upper_bound(100).description('A float as some have decimals')
        )

        name = String()
        sex = String().accepted_values(['male', 'female'])
        survived = Bool().description('If the passenger survived')

        sibsp = (
            Int32()
            .lower_bound(0, is_inclusive=True)
            .upper_bound(20, is_inclusive=True)
            .description('Number of siblings on titanic')
        )

        cabin = String()

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        is_mr = name.contains('Mr.')

    return TitanicPassenger()


@pytest.fixture
def titanic_model(titanic_feature_view: FeatureView) -> ModelContract:
    class Titanic(ModelContract):

        features = titanic_feature_view

        metadata = ModelContract.metadata_with(
            'titanic',
            'A model predicting if a passenger will survive',
            features=[
                features.age,  # type: ignore
                features.sibsp,  # type: ignore
                features.has_siblings,  # type: ignore
                features.is_male,  # type: ignore
                features.is_mr,  # type: ignore
            ],
        )

        will_survive = features.survived.as_classification_target()  # type: ignore

    return Titanic()


@pytest.fixture
def titanic_feature_view_parquet(titanic_source_parquet: ParquetFileSource) -> FeatureView:
    class TitanicPassenger(FeatureView):

        metadata = FeatureViewMetadata(
            name='titanic_parquet',
            description='Some features from the titanic dataset',
            batch_source=titanic_source_parquet,
        )

        passenger_id = Entity(dtype=Int32())

        # Input values
        age = (
            Float().is_required().lower_bound(0).upper_bound(100).description('A float as some have decimals')
        )

        name = String()
        sex = String().accepted_values(['male', 'female'])
        survived = Bool().description('If the passenger survived')

        sibsp = (
            Int32()
            .lower_bound(0, is_inclusive=True)
            .upper_bound(20, is_inclusive=True)
            .description('Number of siblings on titanic')
        )

        cabin = String()

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        is_mr = name.contains('Mr.')

    return TitanicPassenger()


@pytest_asyncio.fixture
async def titanic_feature_store(
    titanic_feature_view: FeatureView, titanic_feature_view_parquet: FeatureView, titanic_model: ModelContract
) -> FeatureStore:
    feature_store = FeatureStore.experimental()
    feature_store.add_feature_view(titanic_feature_view)
    feature_store.add_feature_view(titanic_feature_view_parquet)
    feature_store.add_model(titanic_model)
    return feature_store


@pytest.fixture
def alot_of_transforations_feature_view(titanic_source: CsvFileSource) -> FeatureView:
    class TitanicPassenger(FeatureView):

        metadata = FeatureViewMetadata(
            name='titanic', description='Some features from the titanic dataset', batch_source=titanic_source
        )

        passenger_id = Entity(dtype=Int32())

        # Input values
        age = Float()
        name = String()
        sex = String()
        survived = Bool()
        sibsp = Int32()
        cabin = String().fill_na('Nada')

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        ordinal_sex = sex.ordinal_categories(['male', 'female'])
        filled_age = age.fill_na(0)
        is_mr = name.contains('Mr.')

        adding = sibsp + age
        subtracting = sibsp - age
        floored_age = floor(age)
        ceiled_age = ceil(age)
        rounded_age = round(age)

        inverted_is_mr = ~is_mr
        logical_and = is_mr & survived
        logical_or = is_mr | survived

    return TitanicPassenger()


@pytest_asyncio.fixture
async def alot_of_transforation_feature_store(
    alot_of_transforations_feature_view: FeatureView,
) -> FeatureStore:
    feature_store = FeatureStore.experimental()
    feature_store.add_feature_view(alot_of_transforations_feature_view)
    return feature_store


@pytest_asyncio.fixture
async def combined_view(
    titanic_feature_view: FeatureView, breast_scan_feature_viewout_with_datetime: FeatureView
) -> CombinedFeatureView:
    class SomeCombinedView(CombinedFeatureView):

        metadata = CombinedFeatureViewMetadata(
            name='combined', description='Some features that depend on multiple view'
        )

        titanic = titanic_feature_view
        cancer_scan = breast_scan_feature_viewout_with_datetime

        some_feature = titanic.age + cancer_scan.radius_mean  # type: ignore
        other_feature = titanic.sibsp + cancer_scan.radius_mean  # type: ignore

    return SomeCombinedView()


@pytest_asyncio.fixture
async def combined_feature_store(
    titanic_feature_view: FeatureView,
    breast_scan_feature_viewout_with_datetime: FeatureView,
    combined_view: CombinedFeatureView,
) -> FeatureStore:
    feature_store = FeatureStore.experimental()
    feature_store.add_feature_view(titanic_feature_view)
    feature_store.add_feature_view(breast_scan_feature_viewout_with_datetime)
    feature_store.add_combined_feature_view(combined_view)
    return feature_store


@pytest.fixture
def titanic_feature_view_scd(titanic_source_scd: CsvFileSource) -> FeatureView:
    redis = RedisConfig.localhost()

    class TitanicPassenger(FeatureView):

        metadata = FeatureViewMetadata(
            name='titanic',
            description='Some features from the titanic dataset',
            batch_source=titanic_source_scd,
            stream_source=redis.stream(topic='titanic_stream').with_coder(JsonRecordCoder('json')),
        )

        passenger_id = Entity(dtype=Int32())

        # Input values
        age = (
            Float().is_required().lower_bound(0).upper_bound(100).description('A float as some have decimals')
        )
        updated_at = EventTimestamp()

        sex = String().accepted_values(['male', 'female'])
        survived = Bool().description('If the passenger survived')

        name = String()
        name_embedding = name.embedding(TextVectoriserModel.gensim('glove-wiki-gigaword-50')).indexed(
            embedding_size=50, storage=redis.index(name='name_embedding_index'), metadata=[age, sex]
        )

        sibsp = (
            Int32()
            .lower_bound(0, is_inclusive=True)
            .upper_bound(20, is_inclusive=True)
            .description('Number of siblings on titanic')
        )

        double_sibsp = sibsp * 2
        square_sibsp = sibsp * sibsp

        cabin = String()

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        is_mr = name.contains('Mr.')

    return TitanicPassenger()


@pytest.fixture
def titanic_model_scd(titanic_feature_view_scd: FeatureView) -> ModelContract:
    class Titanic(ModelContract):

        features = titanic_feature_view_scd

        metadata = ModelContract.metadata_with(
            'titanic',
            'A model predicting if a passenger will survive',
            features=[features.age, features.sibsp, features.has_siblings, features.is_male],  # type: ignore
        )

        will_survive = features.survived.as_classification_target()  # type: ignore
        probability = will_survive.probability_of(True)

    return Titanic()


@pytest_asyncio.fixture
async def titanic_feature_store_scd(
    titanic_feature_view_scd: FeatureView,
    titanic_feature_view_parquet: FeatureView,
    titanic_model_scd: ModelContract,
) -> FeatureStore:
    feature_store = FeatureStore.experimental()
    feature_store.add_feature_view(titanic_feature_view_scd)
    feature_store.add_feature_view(titanic_feature_view_parquet)
    feature_store.add_model(titanic_model_scd)
    return feature_store


@dataclass
class FeatureData:
    data: pl.DataFrame
    view: FeatureView


@dataclass
class DataTest:
    sources: list[FeatureData]
    entities: pl.DataFrame
    feature_reference: list[str]
    expected_output: pl.DataFrame


@pytest.fixture
def point_in_time_data_test() -> DataTest:
    from datetime import datetime, timezone

    placeholder_ds = FileSource.parquet_at('placeholder')

    class CreditHistory(FeatureView):

        metadata = FeatureView.metadata_with('credit_history', description='', batch_source=placeholder_ds)

        dob_ssn = String().as_entity()
        event_timestamp = EventTimestamp()
        credit_card_due = Int64()
        student_loan_due = Int64()

        due_sum = credit_card_due + student_loan_due

        bankruptcies = Int32()

    class CreditHistoryAggregate(FeatureView):

        metadata = FeatureView.metadata_with(
            'credit_history_agg', description='', batch_source=placeholder_ds
        )

        dob_ssn = String().as_entity()
        event_timestamp = EventTimestamp()
        credit_card_due = Int64()

        credit_sum = credit_card_due.aggregate().over(weeks=1).sum()

    class Loan(FeatureView):

        metadata = FeatureView.metadata_with('loan', description='', batch_source=placeholder_ds)

        loan_id = Int32().as_entity()
        event_timestamp = EventTimestamp()
        loan_status = Bool().description('If the loan was granted or not')
        personal_income = Int64()
        loan_amount = Int64()

    first_event_timestamp = datetime(2020, 4, 26, 18, 1, 4, 746575, tzinfo=timezone.utc)
    second_event_timestamp = datetime(2020, 4, 27, 18, 1, 4, 746575, tzinfo=timezone.utc)

    credit_data = pl.DataFrame(
        {
            'dob_ssn': [
                '19530219_5179',
                '19520816_8737',
                '19860413_2537',
                '19530219_5179',
                '19520816_8737',
                '19860413_2537',
            ],
            'event_timestamp': [
                first_event_timestamp,
                first_event_timestamp,
                first_event_timestamp,
                second_event_timestamp,
                second_event_timestamp,
                second_event_timestamp,
            ],
            'credit_card_due': [8419, 2944, 833, 5936, 1575, 6263],
            'student_loan_due': [22328, 2515, 33000, 48955, 9501, 35510],
            'bankruptcies': [0, 0, 0, 0, 0, 0],
        }
    )

    loan_data = pl.DataFrame(
        {
            'loan_id': [10000, 10001, 10002, 10000, 10001, 10002],
            'event_timestamp': [
                first_event_timestamp,
                first_event_timestamp,
                first_event_timestamp,
                second_event_timestamp,
                second_event_timestamp,
                second_event_timestamp,
            ],
            'loan_status': [1, 0, 1, 1, 1, 1],
            'personal_income': [59000, 9600, 9600, 65500, 54400, 9900],
            'loan_amount': [35000, 1000, 5500, 35000, 35000, 2500],
        }
    )

    entities = pl.DataFrame(
        {
            'dob_ssn': ['19530219_5179', '19520816_8737', '19860413_2537'],
            'loan_id': [10000, 10001, 10002],
            'event_timestamp': [first_event_timestamp, first_event_timestamp, second_event_timestamp],
        }
    )

    expected_output = pl.DataFrame(
        {
            'dob_ssn': ['19530219_5179', '19520816_8737', '19860413_2537'],
            'loan_id': [10000, 10001, 10002],
            'event_timestamp': [first_event_timestamp, first_event_timestamp, second_event_timestamp],
            'credit_card_due': [8419, 2944, 6263],
            'credit_sum': [8419, 2944, 833 + 6263],
            'student_loan_due': [22328, 2515, 35510],
            'due_sum': [22328 + 8419, 2515 + 2944, 35510 + 6263],
            'personal_income': [59000, 9600, 9900],
        }
    )

    return DataTest(
        sources=[
            FeatureData(data=credit_data, view=CreditHistory()),
            FeatureData(data=loan_data, view=Loan()),
            FeatureData(data=credit_data, view=CreditHistoryAggregate()),
        ],
        entities=entities,
        feature_reference=[
            'credit_history:credit_card_due',
            'credit_history:student_loan_due',
            'credit_history:due_sum',
            'credit_history_agg:credit_sum',
            'loan:personal_income',
        ],
        expected_output=expected_output,
    )
