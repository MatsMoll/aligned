from aligned import Bool, FeatureView, FileSource, Int32, Model, String
from aligned.schemas.feature import FeatureLocation


class View(FeatureView):

    metadata = FeatureView.metedata_with('view', 'test', FileSource.csv_at(''))

    view_id = Int32().as_entity()

    feature_a = String()


class OtherView(FeatureView):

    metadata = FeatureView.metedata_with('other', 'test', FileSource.csv_at(''))

    other_id = Int32().as_entity()

    feature_b = Int32()
    is_true = Bool()


class First(Model):

    view = View()
    other = OtherView()

    metadata = Model.metadata_with('test_model', '', features=[view.feature_a, other.feature_b])

    target = other.is_true.as_target()


class Second(Model):

    first = First()

    metadata = Model.metadata_with('second_model', '', features=[first.target])


def test_model_referenced_as_feature() -> None:
    model = Second.compile()

    feature = list(model.features)[0]

    assert feature.location == FeatureLocation.model('test_model')
    assert feature.name == 'target'
    assert len(model.predictions_view.entities) == 2
