from aligned import feature_view, String, Int32, FileSource
from aligned.schemas.feature import FeatureLocation


def test_feature_view_wrapper_feature_references() -> None:
    @feature_view(name='test', source=FileSource.csv_at('some_file.csv'))
    class Test:

        some_id = Int32().as_entity()

        feature = String()

    NewTest = Test.filter('new_test', where=lambda view: view.feature == 'test')  # type: ignore

    new_test = NewTest()
    test = Test()

    assert new_test.feature._location == FeatureLocation.feature_view('new_test')
    assert test.feature._location == FeatureLocation.feature_view('test')
    assert new_test.some_id._location != test.some_id._location
