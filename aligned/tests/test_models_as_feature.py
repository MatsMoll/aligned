from aligned import Bool, FeatureStore, FileSource, Int32, String
from aligned.feature_view.feature_view import feature_view
from aligned.compiler.model import FeatureInputVersions, model_contract
from aligned.schemas.feature import FeatureLocation


@feature_view('view', FileSource.csv_at(''), 'test')
class View:

    view_id = Int32().as_entity()

    feature_a = String()


@feature_view('other', FileSource.csv_at(''), 'test')
class OtherView:

    other_id = Int32().as_entity()

    feature_b = Int32()
    is_true = Bool()


view = View()
other = OtherView()


@model_contract(
    'test_model',
    features=FeatureInputVersions(
        default_version='v1',
        versions={
            'v1': [view.feature_a, other.feature_b],
            'v2': [view.feature_a, other.feature_b, other.is_true],
        },
    ),
)
class First:

    target = other.is_true.as_classification_label()


first = First()


@model_contract('second_model', features=[first.target])
class Second:
    other_id = Int32().as_entity()
    view_id = Int32().as_entity()


def test_model_referenced_as_feature() -> None:
    model = Second.compile()  # type: ignore

    feature = model.features.default_features[0]

    assert feature.location == FeatureLocation.model('test_model')
    assert feature.name == 'target'
    assert len(model.predictions_view.entities) == 2


def test_model_request() -> None:
    store = FeatureStore.experimental()
    store.add_feature_view(View)  # type: ignore
    store.add_feature_view(OtherView)  # type: ignore
    store.add_model(First)

    assert len(store.feature_views) == 2

    model_request = store.model('test_model').request()
    assert model_request.features_to_include == {'feature_a', 'feature_b', 'view_id', 'other_id'}


def test_model_version() -> None:
    store = FeatureStore.experimental()
    store.add_feature_view(View)  # type: ignore
    store.add_feature_view(OtherView)  # type: ignore
    store.add_model(First)

    assert len(store.feature_views) == 2

    model_request = store.model('test_model').using_version('v2').request()
    assert model_request.features_to_include == {'feature_a', 'is_true', 'feature_b', 'view_id', 'other_id'}
