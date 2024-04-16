import pytest
from aligned import Bool, ContractStore, FileSource, Int32, String
from aligned.feature_view.feature_view import feature_view
from aligned.compiler.model import FeatureInputVersions, model_contract
from aligned.schemas.feature import FeatureLocation


@feature_view(source=FileSource.csv_at(''))
class View:

    view_id = Int32().as_entity()

    feature_a = String()


@feature_view(source=FileSource.csv_at(''))
class OtherView:

    other_id = Int32().as_entity()

    feature_b = Int32()
    is_true = Bool()


view = View()
other = OtherView()


@model_contract(
    name='test_model',
    input_features=FeatureInputVersions(
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


@model_contract(name='first_with_versions', input_features=[view.feature_a, other.feature_b])
class FirstWithVersions:
    some_id = Int32().as_entity()
    target = other.is_true.as_classification_label()
    model_version = String().as_model_version()


@model_contract(name='second_model', input_features=[first.target])
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
    store = ContractStore.experimental()
    store.add_feature_view(View)  # type: ignore
    store.add_feature_view(OtherView)  # type: ignore
    store.add_model(First)

    assert len(store.feature_views) == 2

    model_request = store.model('test_model').request()
    assert model_request.features_to_include == {'feature_a', 'feature_b', 'view_id', 'other_id'}


def test_model_version() -> None:
    store = ContractStore.experimental()
    store.add_feature_view(View)  # type: ignore
    store.add_feature_view(OtherView)  # type: ignore
    store.add_model(First)

    assert len(store.feature_views) == 2

    model_request = store.model('test_model').using_version('v2').request()
    assert model_request.features_to_include == {'feature_a', 'is_true', 'feature_b', 'view_id', 'other_id'}


@pytest.mark.asyncio
async def test_load_preds_with_different_model_version() -> None:
    import polars as pl

    store = ContractStore.experimental()
    store.add_model(FirstWithVersions)

    source = FileSource.csv_at('test_data/model_preds.csv')

    await source.write_polars(
        pl.DataFrame(
            {'some_id': [1, 2, 1, 2], 'target': [0, 1, 1, 1], 'model_version': ['v1', 'v1', 'v2', 'v2']}
        ).lazy()
    )

    model_store = store.model('first_with_versions').using_source(source)

    df = await model_store.predictions_for({'some_id': [1, 2], 'model_version': ['v2', 'v2']}).to_polars()

    assert df['target'].to_list() == [False, True]

    new_df = await model_store.predictions_for(
        {'some_id': [1, 2], 'model_version': ['v2', 'v2']}, model_version_as_entity=True
    ).to_polars()

    assert new_df['target'].to_list() == [True, True]
