from dataclasses import dataclass
from datetime import timedelta
from math import ceil, floor

import polars as pl
import pytest

from aligned import (
    Bool,
    EventTimestamp,
    FileSource,
    Float32,
    Int32,
    Int64,
    RedisConfig,
    String,
    Int8,
    EmbeddingModel,
    feature_view,
)
from aligned.feature_view.feature_view import FeatureView, FeatureViewWrapper
from aligned.compiler.model import model_contract, ModelContractWrapper
from aligned.feature_store import ContractStore
from aligned.retrival_job import DerivedFeatureJob, RetrivalJob, RetrivalRequest
from aligned.schemas.date_formatter import DateFormatter
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReference, FeatureType
from aligned.schemas.record_coders import JsonRecordCoder
from aligned.sources.local import CsvFileSource, FileFullJob, LiteralReference, ParquetFileSource


@pytest.fixture
def retrival_request_without_derived() -> RetrivalRequest:
    return RetrivalRequest(
        name='test',
        location=FeatureLocation.feature_view('test'),
        entities={Feature(name='id', dtype=FeatureType.int32())},
        features={
            Feature(name='a', dtype=FeatureType.int32()),
            Feature(name='b', dtype=FeatureType.int32()),
        },
        derived_features=set(),
        event_timestamp=None,
    )


@pytest.fixture
def retrival_job(retrival_request_without_derived: RetrivalRequest) -> RetrivalJob:

    return FileFullJob(
        LiteralReference(pl.DataFrame({'id': [1, 2, 3, 4, 5], 'a': [3, 4, 2, 3, 4], 'b': [1, 1, 1, 2, 4]})),
        request=retrival_request_without_derived,
    )


@pytest.fixture
def retrival_request_with_derived() -> RetrivalRequest:
    from aligned.schemas.feature import EventTimestamp as TimestampFeature
    from aligned.schemas.transformation import Addition

    return RetrivalRequest(
        name='test_with_ts',
        location=FeatureLocation.feature_view('test_with_ts'),
        entities={Feature(name='id', dtype=FeatureType.int32())},
        features={
            Feature(name='c', dtype=FeatureType.int32()),
            Feature(name='d', dtype=FeatureType.int32()),
        },
        derived_features={
            DerivedFeature(
                name='c+d',
                dtype=FeatureType.int32(),
                depending_on={
                    FeatureReference(
                        name='c',
                        location=FeatureLocation.feature_view('test_with_ts'),
                    ),
                    FeatureReference(
                        name='d',
                        location=FeatureLocation.feature_view('test_with_ts'),
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

    date = datetime(year=2022, month=1, day=1)
    one_day = timedelta(days=1)
    return DerivedFeatureJob(
        job=FileFullJob(
            LiteralReference(
                pl.DataFrame(
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
        entities={Feature(name='id', dtype=FeatureType.int32())},
        features=set(),
        derived_features={
            DerivedFeature(
                name='a+c+d',
                dtype=FeatureType.int32(),
                depending_on={
                    FeatureReference(
                        name='c+d',
                        location=FeatureLocation.feature_view('test_with_ts'),
                    ),
                    FeatureReference(name='a', location=FeatureLocation.feature_view('test')),
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
def breast_scan_feature_viewout_with_datetime(scan_without_datetime: CsvFileSource) -> FeatureViewWrapper:
    @feature_view(
        name='breast_features',
        description='Features defining a scan and diagnose of potential cancer cells',
        source=scan_without_datetime,
    )
    class BreastDiagnoseFeatureView:

        scan_id = Int32().as_entity()
        diagnosis = String().description('The given diagnose. M for malignant, and B for benigne')
        is_malignant = (diagnosis == 'M').description('If the scanned cells was diagnosed as dangerous')

        radius_mean = Float32()
        radius_se = Float32()
        radius_worst = Float32()

        texture_mean = Float32()
        texture_se = Float32()
        texture_worst = Float32()

        perimeter_mean = Float32()
        perimeter_se = Float32()
        perimeter_worst = Float32()

        area_mean = Float32()
        area_se = Float32()
        area_worst = Float32()

        smoothness_mean = Float32()
        smoothness_se = Float32()
        smoothness_worst = Float32()

        compactness_mean = Float32()
        compactness_se = Float32()
        compactness_worst = Float32()

        concavity_mean = Float32()
        concavity_se = Float32()
        concavity_worst = Float32()

        concave_points_mean = Float32()
        concave_points_se = Float32()
        concave_points_worst = Float32()

        symmetry_mean = Float32()
        symmetry_se = Float32()
        symmetry_worst = Float32()

        fractal_dimension_mean = Float32()
        fractal_dimension_se = Float32()
        fractal_dimension_worst = Float32()

    return BreastDiagnoseFeatureView


@pytest.fixture
def breast_scan_without_timestamp_feature_store(
    breast_scan_feature_viewout_with_datetime: FeatureView,
) -> ContractStore:
    store = ContractStore.empty()
    store.add_feature_view(breast_scan_feature_viewout_with_datetime)
    return store


@pytest.fixture
def scan_with_datetime() -> CsvFileSource:
    return FileSource.csv_at(
        path='test_data/data-with-datetime.csv',
        date_formatter=DateFormatter.string_format('%Y-%m-%d %H:%M:%S'),
    )


@pytest.fixture
def breast_scan_feature_view_with_datetime(scan_with_datetime: CsvFileSource) -> FeatureViewWrapper:
    @feature_view(
        name='breast_features',
        description='Features defining a scan and diagnose of potential cancer cells',
        source=scan_with_datetime,
    )
    class BreastDiagnoseFeatureView:

        scan_id = Int32().as_entity()

        created_at = EventTimestamp()

        diagnosis = String().description('The given diagnose. M for malignant, and B for benigne')
        is_malignant = (diagnosis == 'M').description('If the scanned cells was diagnosed as dangerous')

        radius_mean = Float32()
        radius_se = Float32()
        radius_worst = Float32()

        texture_mean = Float32()
        texture_se = Float32()
        texture_worst = Float32()

        perimeter_mean = Float32()
        perimeter_se = Float32()
        perimeter_worst = Float32()

        area_mean = Float32()
        area_se = Float32()
        area_worst = Float32()

        smoothness_mean = Float32()
        smoothness_se = Float32()
        smoothness_worst = Float32()

        compactness_mean = Float32()
        compactness_se = Float32()
        compactness_worst = Float32()

        concavity_mean = Float32()
        concavity_se = Float32()
        concavity_worst = Float32()

        concave_points_mean = Float32()
        concave_points_se = Float32()
        concave_points_worst = Float32()

        symmetry_mean = Float32()
        symmetry_se = Float32()
        symmetry_worst = Float32()

        fractal_dimension_mean = Float32()
        fractal_dimension_se = Float32()
        fractal_dimension_worst = Float32()

    return BreastDiagnoseFeatureView


@pytest.fixture
def breast_scan_feature_view_with_datetime_and_aggregation(
    scan_with_datetime: CsvFileSource,
) -> FeatureViewWrapper:
    @feature_view(
        name='breast_features',
        description='Features defining a scan and diagnose of potential cancer cells',
        source=scan_with_datetime,
    )
    class BreastDiagnoseFeatureView:

        scan_id = Int32().as_entity()

        created_at = EventTimestamp()

        diagnosis = String().description('The given diagnose. M for malignant, and B for benigne')
        is_malignant = (diagnosis == 'M').description('If the scanned cells was diagnosed as dangerous')

        radius_mean = Float32()
        radius_se = Float32()
        radius_worst = Float32()

        texture_mean = Float32()
        texture_se = Float32()
        texture_worst = Float32()

        perimeter_mean = Float32()
        perimeter_se = Float32()
        perimeter_worst = Float32()

        area_mean = Float32()
        area_se = Float32()
        area_worst = Float32()

        smoothness_mean = Float32()
        smoothness_se = Float32()
        smoothness_worst = Float32()

        compactness_mean = Float32()
        compactness_se = Float32()
        compactness_worst = Float32()

        concavity_mean = Float32()
        concavity_se = Float32()
        concavity_worst = Float32()

        concave_points_mean = Float32()
        concave_points_se = Float32()
        concave_points_worst = Float32()

        symmetry_mean = Float32()
        symmetry_se = Float32()
        symmetry_worst = Float32()

        fractal_dimension_mean = Float32()
        fractal_dimension_se = Float32()
        fractal_dimension_worst = Float32()

    return BreastDiagnoseFeatureView


@pytest.fixture
def breast_scan_with_timestamp_feature_store(
    breast_scan_feature_view_with_datetime: FeatureView,
) -> ContractStore:
    store = ContractStore.empty()
    store.add_feature_view(breast_scan_feature_view_with_datetime)
    return store


@pytest.fixture
def breast_scan_with_timestamp_and_aggregation_feature_store(
    breast_scan_feature_view_with_datetime_and_aggregation: FeatureView,
) -> ContractStore:
    store = ContractStore.empty()
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
def titanic_source_parquet() -> ParquetFileSource:
    return FileSource.parquet_at('test_data/titanic.parquet')


@pytest.fixture
def titanic_feature_view(titanic_source: CsvFileSource) -> FeatureViewWrapper:
    @feature_view(name='titanic', description='Some features from the titanic dataset', source=titanic_source)
    class TitanicPassenger:

        passenger_id = Int32().as_entity()

        # Input values
        age = Float32().lower_bound(0).upper_bound(100).description('A float as some have decimals')

        name = String().is_optional()
        sex = String().is_optional().accepted_values(['male', 'female'])
        survived = Int8().description('If the passenger survived')

        sibsp = (
            Int32().is_optional().lower_bound(0).upper_bound(20).description('Number of siblings on titanic')
        )

        cabin = String().is_optional()

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        is_mr = name.contains('Mr.')

    return TitanicPassenger


@pytest.fixture
def titanic_model(titanic_feature_view: FeatureViewWrapper) -> ModelContractWrapper:

    features = titanic_feature_view()

    @model_contract(
        name='titanic',
        description='A model predicting if a passenger will survive',
        input_features=[
            features.age,  # type: ignore
            features.sibsp,  # type: ignore
            features.has_siblings,  # type: ignore
            features.is_male,  # type: ignore
            features.is_mr,  # type: ignore
        ],
    )
    class Titanic:
        passenger_id = Int32().as_entity()

        will_survive = features.survived.as_classification_label()  # type: ignore

    return Titanic


@pytest.fixture
def titanic_feature_view_parquet(titanic_source_parquet: ParquetFileSource) -> FeatureViewWrapper:
    @feature_view(
        name='titanic_parquet',
        description='Some features from the titanic dataset',
        source=titanic_source_parquet,
    )
    class TitanicPassenger:
        passenger_id = Int32().as_entity()

        # Input values
        age = (
            Float32()
            .is_required()
            .lower_bound(0)
            .upper_bound(100)
            .description('A float as some have decimals')
        )

        name = String()
        sex = String().accepted_values(['male', 'female'])
        survived = Bool().description('If the passenger survived')

        sibsp = Int32().lower_bound(0).upper_bound(20).description('Number of siblings on titanic')

        cabin = String()

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        is_mr = name.contains('Mr.')

    return TitanicPassenger


@pytest.fixture
def titanic_feature_store(
    titanic_feature_view: FeatureView,
    titanic_feature_view_parquet: FeatureView,
    titanic_model: ModelContractWrapper,
) -> ContractStore:
    feature_store = ContractStore.empty()
    feature_store.add_feature_view(titanic_feature_view)
    feature_store.add_feature_view(titanic_feature_view_parquet)
    feature_store.add_model(titanic_model)
    return feature_store


@pytest.fixture
def alot_of_transforations_feature_view(titanic_source: CsvFileSource) -> FeatureViewWrapper:
    @feature_view(name='titanic', description='Some features from the titanic dataset', source=titanic_source)
    class TitanicPassenger:

        passenger_id = Int32().as_entity()

        # Input values
        age = Float32()
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

    return TitanicPassenger


@pytest.fixture
def alot_of_transforation_feature_store(
    alot_of_transforations_feature_view: FeatureViewWrapper,
) -> ContractStore:
    feature_store = ContractStore.empty()
    feature_store.add_feature_view(alot_of_transforations_feature_view)
    return feature_store


@pytest.fixture
def combined_feature_store(
    titanic_feature_view: FeatureViewWrapper,
    breast_scan_feature_viewout_with_datetime: FeatureViewWrapper,
) -> ContractStore:
    feature_store = ContractStore.empty()
    feature_store.add_feature_view(titanic_feature_view)
    feature_store.add_feature_view(breast_scan_feature_viewout_with_datetime)
    return feature_store


@pytest.fixture
def titanic_feature_view_scd(titanic_source_scd: CsvFileSource) -> FeatureViewWrapper:
    redis = RedisConfig.localhost()

    @feature_view(
        name='titanic',
        description='Some features from the titanic dataset',
        source=titanic_source_scd,
        stream_source=redis.stream(topic='titanic_stream').with_coder(JsonRecordCoder('json')),
    )
    class TitanicPassenger:

        passenger_id = Int32().as_entity()

        # Input values
        age = (
            Float32()
            .is_required()
            .lower_bound(0)
            .upper_bound(100)
            .description('A float as some have decimals')
        )
        updated_at = EventTimestamp()

        sex = String().accepted_values(['male', 'female'])
        survived = Bool().description('If the passenger survived')

        name = String()
        name_embedding = name.embedding(EmbeddingModel.gensim('glove-wiki-gigaword-50')).indexed(
            embedding_size=50, storage=redis.index(name='name_embedding_index'), metadata=[age, sex]
        )

        sibsp = Int32().lower_bound(0).upper_bound(20).description('Number of siblings on titanic')

        double_sibsp = sibsp * 2
        square_sibsp = sibsp * sibsp

        cabin = String()

        # Transformed features
        has_siblings = sibsp != 0
        is_male, is_female = sex.one_hot_encode(['male', 'female'])
        is_mr = name.contains('Mr.')

    return TitanicPassenger


@pytest.fixture
def titanic_model_scd(titanic_feature_view_scd: FeatureViewWrapper) -> ModelContractWrapper:

    features = titanic_feature_view_scd()

    @model_contract(
        name='titanic',
        description='A model predicting if a passenger will survive',
        input_features=[
            features.age,  # type: ignore
            features.sibsp,  # type: ignore
            features.has_siblings,  # type: ignore
            features.is_male,  # type: ignore
        ],
        acceptable_freshness=timedelta(days=1),
        unacceptable_freshness=timedelta(days=2),
    )
    class Titanic:

        will_survive = features.survived.as_classification_label()  # type: ignore
        probability = will_survive.probability_of(True)

    return Titanic


@pytest.fixture
def titanic_feature_store_scd(
    titanic_feature_view_scd: FeatureViewWrapper,
    titanic_feature_view_parquet: FeatureViewWrapper,
    titanic_model_scd: ModelContractWrapper,
) -> ContractStore:
    feature_store = ContractStore.empty()
    feature_store.add_feature_view(titanic_feature_view_scd)
    feature_store.add_feature_view(titanic_feature_view_parquet)
    feature_store.add_model(titanic_model_scd)
    return feature_store


@dataclass
class FeatureData:
    data: pl.DataFrame
    view: FeatureViewWrapper


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

    @feature_view(name='credit_history', source=placeholder_ds)
    class CreditHistory:

        dob_ssn = String().as_entity()
        event_timestamp = EventTimestamp()
        credit_card_due = Int64()
        student_loan_due = Int64()

        due_sum = credit_card_due + student_loan_due

        bankruptcies = Int32()

    @feature_view(name='credit_history_agg', source=placeholder_ds)
    class CreditHistoryAggregate:
        dob_ssn = String().as_entity()
        event_timestamp = EventTimestamp()
        credit_card_due = Int64()

        credit_sum = credit_card_due.aggregate().over(weeks=1).sum()

    @feature_view(name='loan', source=placeholder_ds)
    class Loan:

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
            FeatureData(data=credit_data, view=CreditHistory),
            FeatureData(data=loan_data, view=Loan),
            FeatureData(data=credit_data, view=CreditHistoryAggregate),
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


@pytest.fixture
def point_in_time_data_test_wituout_event_timestamp() -> DataTest:
    from datetime import datetime, timezone

    placeholder_ds = FileSource.parquet_at('placeholder')

    @feature_view(name='credit_history', description='', source=placeholder_ds)
    class CreditHistory:

        dob_ssn = String().as_entity()
        event_timestamp = EventTimestamp()
        credit_card_due = Int64()
        student_loan_due = Int64()

        due_sum = credit_card_due + student_loan_due

        bankruptcies = Int32()

    @feature_view(name='credit_history_agg', description='', source=placeholder_ds)
    class CreditHistoryAggregate:
        dob_ssn = String().as_entity()
        event_timestamp = EventTimestamp()
        credit_card_due = Int64()

        credit_sum = credit_card_due.aggregate().over(weeks=1).sum()

    @feature_view(name='loan', description='', source=placeholder_ds)
    class Loan:

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
        }
    )

    expected_output = pl.DataFrame(
        {
            'dob_ssn': ['19530219_5179', '19520816_8737', '19860413_2537'],
            'loan_id': [10000, 10001, 10002],
            'credit_card_due': [5936, 1575, 6263],
            # 'credit_sum': [8419 + 5936, 2944 + 1575, 833 + 6263],
            'student_loan_due': [48955, 9501, 35510],
            'due_sum': [5936 + 48955, 1575 + 9501, 6263 + 35510],
            'personal_income': [65500, 54400, 9900],
        }
    )

    return DataTest(
        sources=[
            FeatureData(data=credit_data, view=CreditHistory),
            FeatureData(data=loan_data, view=Loan),
            FeatureData(data=credit_data, view=CreditHistoryAggregate),
        ],
        entities=entities,
        feature_reference=[
            'credit_history:credit_card_due',
            'credit_history:student_loan_due',
            'credit_history:due_sum',
            'loan:personal_income',
        ],
        expected_output=expected_output,
    )
