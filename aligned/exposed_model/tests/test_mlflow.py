from contextlib import suppress
import pytest
from aligned import ExposedModel, model_contract, String, Int32, EventTimestamp, feature_view, FileSource


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
