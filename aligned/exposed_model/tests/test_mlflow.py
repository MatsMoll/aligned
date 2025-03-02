from os import environ
import polars as pl
from typing import TYPE_CHECKING
import pytest
from aligned import (
    feature_view,
    model_contract,
    String,
    Int32,
    FileSource,
    ContractStore,
)

from aligned.lazy_imports import mlflow, pandas as pd
from aligned.exposed_model.interface import python_function
from aligned.exposed_model.mlflow import (
    signature_for_features,
    reference_metadata_for_features,
    in_memory_mlflow,
    signature_for_contract,
)

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


@feature_view(
    name="input",
    source=FileSource.parquet_at("non-existing-data"),
)
class InputFeatureView:
    entity_id = String().as_entity()
    x = Int32()
    other = Int32()


input = InputFeatureView()


@model_contract(
    input_features=[InputFeatureView().x],
    exposed_model=python_function(lambda df: df["x"] * 2),
)
class MyModelContract:
    entity_id = String().as_entity()

    prediction = input.x.as_regression_target()


@model_contract(
    input_features=[InputFeatureView().x],
    exposed_model=in_memory_mlflow("custom-model"),
)
class MyModelContract2:
    entity_id = String().as_entity()
    other_pred = input.other.as_regression_target()


class CustomModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: pd.DataFrame, params=None):
        return (model_input * 2).sum(axis=1, numeric_only=True)


@pytest.mark.asyncio
@pytest.mark.skipif(
    environ.get("MLFLOW_REGISTRY_URI") is None,
    reason="Only runs if an MLFlow instance is up and running",
)
async def test_mlflow_referencing():
    store = ContractStore.empty()
    store.add(InputFeatureView)
    store.add(MyModelContract)
    store.add(MyModelContract2)

    expected_results = [2, 6, 4]

    model = CustomModel()
    model_name = "custom-model"
    model_store = store.model(MyModelContract2)

    mlflow.start_run()
    model_info: ModelInfo = mlflow.pyfunc.log_model(
        model_name,
        python_model=model,
        signature=signature_for_contract(model_store),
        metadata={
            "feature_refs": reference_metadata_for_features(
                model_store.input_features()
            )
        },
        registered_model_name=model_name,
    )
    mlflow.end_run()

    client = mlflow.MlflowClient()
    assert model_info.registered_model_version

    client.set_registered_model_alias(
        model_name, "champion", str(model_info.registered_model_version)
    )
    df = (
        await store.model(MyModelContract2)
        .predict_over({"entity_id": ["a", "b", "c"], "x": [1, 3, 2]})
        .to_polars()
    )

    assert df.sort("entity_id", descending=False)["other_pred"].equals(
        pl.Series(expected_results)
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    environ.get("MLFLOW_REGISTRY_URI") is None,
    reason="Only runs if an MLFlow instance is up and running",
)
async def test_mlflow_out_of_date_referencing():
    store = ContractStore.empty()
    store.add(InputFeatureView)
    store.add(MyModelContract)
    store.add(MyModelContract2)

    expected_results = [6, 18, 12]

    model = CustomModel()
    model_name = "custom-model"

    signature = signature_for_features(
        inputs=[
            InputFeatureView().x.feature(),
            MyModelContract().prediction.compile().feature,
        ],
        outputs=[MyModelContract2().other_pred.compile().feature],
    )

    mlflow.start_run()
    model_info: ModelInfo = mlflow.pyfunc.log_model(
        model_name,
        python_model=model,
        signature=signature,
        metadata={
            "feature_refs": reference_metadata_for_features(
                [
                    InputFeatureView().x.feature_reference(),
                    MyModelContract().prediction.feature_reference(),
                ]
            )
        },
        registered_model_name=model_name,
    )
    mlflow.end_run()

    client = mlflow.MlflowClient()
    assert model_info.registered_model_version
    client.set_registered_model_alias(
        model_name, "champion", str(model_info.registered_model_version)
    )
    df = (
        await store.model(MyModelContract2)
        .predict_over({"entity_id": ["a", "b", "c"], "x": [1, 3, 2]})
        .to_polars()
    )

    assert df.sort("entity_id", descending=False)["other_pred"].equals(
        pl.Series(expected_results)
    )
