import pytest
from aligned import feature_view, String, Int32, FileSource, Float
from aligned.compiler.feature_factory import EventTimestamp
from aligned.schemas.feature import FeatureLocation


@feature_view(name='test', source=FileSource.csv_at('test_data/test.csv'))
class DefaultValueTest:

    some_id = Int32().as_entity()

    feature = Int32()

    other_value = String().default_value('Hello')
    optional_value = Int32().is_optional()

    other_default = Float().default_value(0)


@feature_view(name='test', source=FileSource.csv_at('some_file.csv'))
class Test:

    some_id = Int32().as_entity()

    feature = String()


@feature_view(name='test', source=FileSource.csv_at('some_file.csv'))
class TestDerived:

    some_id = Int32().as_entity()

    feature = String()

    contains_hello = feature.contains('Hello')


@feature_view(name='test', source=FileSource.csv_at('some_file.csv'))
class TestEventTimestamp:
    some_id = Int32().as_entity()
    feature = String()
    loaded_at = EventTimestamp()
    contains_hello = feature.contains('Hello')


def test_event_timestamp() -> None:
    view = TestEventTimestamp.compile()
    assert len(view.entities) == 1
    assert view.event_timestamp is not None
    assert len(view.features) == 1
    assert len(view.derived_features) == 1


def test_feature_view_wrapper_feature_references() -> None:

    NewTest = Test.filter('new_test', lambda view: view.feature == 'test')  # type: ignore

    new_test = NewTest()
    test = Test()

    assert new_test.feature._location == FeatureLocation.feature_view('new_test')
    assert test.feature._location == FeatureLocation.feature_view('test')
    assert new_test.some_id._location != test.some_id._location


@pytest.mark.asyncio
async def test_feature_view_wrapper_from_data() -> None:

    test_job = Test.from_data({'some_id': [10, 2, 4], 'feature': ['Hello', 'Test', 'World']})

    result = await test_job.to_pandas()
    assert result.shape[0] == 3
    assert result.shape[1] == 2

    test_job = TestDerived.from_data({'some_id': [10, 2, 4], 'feature': ['Hello', 'Test', 'World']})

    result = await test_job.to_pandas()
    assert result.shape[0] == 3
    assert result.shape[1] == 2

    result = await test_job.derive_features().to_pandas()
    assert result.shape[0] == 3
    assert result.shape[1] == 3

    test_invalid_result = Test.drop_invalid({'some_id': ['hello', 10, 2], 'feature': ['Hello', 'test', 2]})

    # Returns two as the int can be casted to a str, but a str can not be casted to int
    assert len(test_invalid_result['some_id']) == 2


@pytest.mark.asyncio
async def test_fill_missing_features() -> None:
    df = await DefaultValueTest.query().all().to_polars()

    fill_columns = ['other_default', 'other_value', 'optional_value']
    for col in fill_columns:
        assert col in df.columns
