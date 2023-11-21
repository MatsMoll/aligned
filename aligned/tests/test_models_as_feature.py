from aligned import Bool, FeatureStore, FileSource, Int32, String
from aligned.feature_view.feature_view import FeatureView
from aligned.compiler.model import ModelContract
from aligned.schemas.feature import FeatureLocation


class View(FeatureView):

    metadata = FeatureView.metadata_with('view', 'test', FileSource.csv_at(''))

    view_id = Int32().as_entity()

    feature_a = String()


class OtherView(FeatureView):

    metadata = FeatureView.metadata_with('other', 'test', FileSource.csv_at(''))

    other_id = Int32().as_entity()

    feature_b = Int32()
    is_true = Bool()


class First(ModelContract):

    view = View()
    other = OtherView()

    metadata = ModelContract.metadata_with('test_model', features=[view.feature_a, other.feature_b])

    target = other.is_true.as_classification_label()


class Second(ModelContract):

    first = First()

    metadata = ModelContract.metadata_with('second_model', features=[first.target])


def test_model_referenced_as_feature() -> None:
    model = Second.compile()

    feature = list(model.features)[0]

    assert feature.location == FeatureLocation.model('test_model')
    assert feature.name == 'target'
    assert len(model.predictions_view.entities) == 2


def test_model_request() -> None:
    store = FeatureStore.experimental()
    store.add_feature_view(View())
    store.add_feature_view(OtherView())
    store.add_model(First())

    assert len(store.feature_views) == 2

    model_request = store.model('test_model').request()
    assert model_request.features_to_include == {'feature_a', 'feature_b', 'view_id', 'other_id'}
