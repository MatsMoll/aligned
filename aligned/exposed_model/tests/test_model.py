from contextlib import suppress
import pytest
from aligned import ExposedModel, model_contract, String, Int32, EventTimestamp, feature_view, FileSource
from aligned.data_source.batch_data_source import DummyDataSource
from aligned.exposed_model.interface import python_function


@pytest.mark.asyncio
async def test_mlflow() -> None:
    from mlflow.tracking import MlflowClient
    import mlflow

    mlflow.set_tracking_uri('test_data/mlruns')

    model_name = 'test_model'
    model_alias = 'Champion'

    mlflow_client = MlflowClient()

    with suppress(mlflow.exceptions.MlflowException):
        mlflow_client.delete_registered_model(model_name)

    def predict(data):
        return data * 2

    mlflow.pyfunc.log_model(
        artifact_path='model', python_model=predict, registered_model_name=model_name, input_example=[1, 2, 3]
    )
    mlflow_client.set_registered_model_alias(name=model_name, alias=model_alias, version=1)  # type: ignore

    @feature_view(
        name='input',
        source=FileSource.parquet_at('non-existing-data'),
    )
    class InputFeatureView:
        entity_id = String().as_entity()
        x = Int32()

    input = InputFeatureView()

    @model_contract(
        input_features=[InputFeatureView().x],
        exposed_model=ExposedModel.in_memory_mlflow(
            model_name=model_name,
            model_alias=model_alias,
        ),
    )
    class MyModelContract:
        entity_id = String().as_entity()
        predicted_at = EventTimestamp()
        prediction = input.x.as_regression_target()
        model_version = String().as_model_version()

    preds = await MyModelContract.predict_over(
        values={'entity_id': ['a', 'b'], 'x': [1, 2]}, needed_views=[InputFeatureView]
    ).to_polars()

    assert preds['prediction'].to_list() == [2, 4]
    assert 'model_version' in preds.columns
    assert 'predicted_at' in preds.columns


@pytest.mark.asyncio
async def test_model() -> None:
    @feature_view(
        name='input',
        source=FileSource.parquet_at('non-existing-data'),
    )
    class InputFeatureView:
        entity_id = String().as_entity()
        x = Int32()
        other = Int32()

    input = InputFeatureView()

    @model_contract(
        input_features=[InputFeatureView().x], exposed_model=python_function(lambda df: df['x'] * 2)
    )
    class MyModelContract:
        entity_id = String().as_entity()

        prediction = input.x.as_regression_target()

    @model_contract(
        input_features=[InputFeatureView().x, MyModelContract().prediction],
        exposed_model=python_function(lambda df: df['prediction'] * 3),
    )
    class MyModelContract2:
        entity_id = String().as_entity()

        other_pred = input.other.as_regression_target()

    entities = {'entity_id': ['a', 'b'], 'x': [1, 2]}
    pred_job = MyModelContract2.predict_over(entities, needed_views=[InputFeatureView, MyModelContract])
    preds = await pred_job.to_polars()

    assert set(pred_job.request_result.feature_columns) == {'x', 'prediction', 'other_pred'}
    assert preds['other_pred'].to_list() == [6, 12]


@pytest.mark.asyncio
async def test_pipeline_model_with_source() -> None:
    @feature_view(
        name='input',
        source=FileSource.parquet_at('non-existing-data'),
    )
    class InputFeatureView:
        entity_id = String().as_entity()
        x = Int32()
        other = Int32()

    input = InputFeatureView()

    @model_contract(
        input_features=[InputFeatureView().x],
        exposed_model=python_function(lambda df: df['x'] * 2),
        output_source=DummyDataSource(),
    )
    class MyModelContract:
        entity_id = String().as_entity()

        prediction = input.x.as_regression_target()

    @model_contract(
        input_features=[InputFeatureView().x, MyModelContract().prediction],
        exposed_model=python_function(lambda df: df['prediction'] * 3),
    )
    class MyModelContract2:
        entity_id = String().as_entity()

        other_pred = input.other.as_regression_target()

    entities = {'entity_id': ['a', 'b'], 'x': [1, 2]}
    preds = await MyModelContract2.predict_over(
        entities, needed_views=[InputFeatureView, MyModelContract]
    ).to_polars()

    assert 'other_pred' in preds
