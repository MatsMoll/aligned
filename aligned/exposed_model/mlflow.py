from __future__ import annotations

import polars as pl
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from aligned.retrival_job import RetrivalJob
from aligned.exposed_model.interface import ExposedModel

from aligned.schemas.feature import Feature, FeatureReference, FeatureLocation, FeatureType

if TYPE_CHECKING:
    from aligned.feature_store import ModelFeatureStore, ContractStore

try:
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, ColSpec, TensorSpec

    def mlflow_spec(feature: Feature, location: FeatureLocation) -> ColSpec | TensorSpec | str:
        dtype = feature.dtype

        ref_id = feature.as_reference(location).identifier

        if dtype.name == 'float':
            return ColSpec('float', name=ref_id)
        elif dtype.name == 'double':
            return ColSpec('double', name=ref_id)
        elif dtype.name == 'string':
            return ColSpec('string', name=ref_id)
        elif dtype.is_numeric:
            return ColSpec('integer', name=ref_id)
        elif dtype.is_datetime:
            return ColSpec('datetime', name=ref_id)
        elif dtype.is_embedding:
            import numpy as np

            return TensorSpec(
                type=np.dtype(np.float32),
                shape=(-1, dtype.embedding_size()),
                name=ref_id,
            )
        return dtype.name

    def signature_for_contract(model_contract_name: str, store: ContractStore) -> ModelSignature:
        model_contract = store.model(model_contract_name)

        input_req = model_contract.request()
        input_reqs = input_req.needed_requests

        output_schema: Schema | None = None
        pred_view = model_contract.model.predictions_view

        try:
            labels = pred_view.labels()
        except Exception:
            labels = None

        if labels:
            output_schema = Schema(
                [
                    mlflow_spec(label, FeatureLocation.model(model_contract_name))  # type: ignore
                    for label in labels
                ]
            )

        all_features = []
        for request in input_reqs:
            for feature in request.returned_features:
                if feature.name not in input_req.features_to_include:
                    continue

                all_features.append(mlflow_spec(feature, request.location))

        return ModelSignature(inputs=Schema(all_features), outputs=output_schema)

except ModuleNotFoundError:
    pass


def in_memory_mlflow(
    model_name: str,
    model_alias: str,
    model_contract_version_tag: str | None = None,
) -> ExposedModel:
    """A model that is loaded from MLFlow using the given model name and alias.

    This will also run in memory, and not require that `mlflow` is installed.
    """
    return InMemMLFlowAlias(
        model_name=model_name,
        model_alias=model_alias,
        model_contract_version_tag=model_contract_version_tag,
    )


def mlflow_server(
    host: str,
    model_alias: str | None = None,
    model_name: str | None = None,
    timeout: int = 30,
) -> ExposedModel:
    """Calls an MLFlow server to get the model predictions.

    This will load the features described in the model signature, and pass them to the server.
    """
    return MLFlowServer(
        host=host,
        model_name=model_name,
        model_alias=model_alias or 'champion',
        timeout=timeout,
    )


@dataclass
class InMemMLFlowAlias(ExposedModel):

    model_name: str
    model_alias: str

    model_contract_version_tag: str | None

    model_type: str = 'latest_mlflow'

    @property
    def exposed_at_url(self) -> str | None:
        return None

    @property
    def as_markdown(self) -> str:
        return f"""Using the latest MLFlow model: `{self.model_name}`."""

    def get_model_version(self):
        from mlflow.tracking import MlflowClient

        mlflow_client = MlflowClient()

        return mlflow_client.get_model_version_by_alias(self.model_name, self.model_alias)

    def contract_version(self, model_version) -> str:
        version = 'default'
        if self.model_contract_version_tag:
            if self.model_contract_version_tag not in model_version.tags:  # noqa
                raise ValueError(
                    f"Model contract version tag {self.model_contract_version_tag} not "
                    'found in model version tags'
                )
            else:
                version = model_version.tags[self.model_contract_version_tag]
        return version

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        mv = self.get_model_version()
        version = self.contract_version(mv)
        return store.feature_references_for(version)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        mv = self.get_model_version()
        version = self.contract_version(mv)
        return store.using_version(version).needed_entities()

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        import mlflow
        import polars as pl
        from datetime import datetime, timezone

        pred_label = list(store.model.predictions_view.labels())[0]
        pred_at = store.model.predictions_view.event_timestamp
        model_version_column = store.model.predictions_view.model_version_column
        mv = None

        if model_version_column:
            mv = self.get_model_version()

        model_uri = f"models:/{self.model_name}@{self.model_alias}"
        mv = self.get_model_version()

        model = mlflow.pyfunc.load_model(model_uri)

        job = store.features_for(values)
        df = await job.to_polars()

        features = job.request_result.feature_columns
        predictions = model.predict(df[features])

        if pred_at:
            df = df.with_columns(
                pl.lit(datetime.now(timezone.utc)).alias(pred_at.name),
            )

        if mv and model_version_column:
            df = df.with_columns(
                pl.lit(mv.version).alias(model_version_column.name),
            )

        return df.with_columns(
            pl.Series(name=pred_label.name, values=predictions),
        )


@dataclass
class MLFlowServer(ExposedModel):
    """
    Describes a model exposed through a mlflow server.

    This also assumes that the model have a signature where each column is a feature reference.
    Meaning on the format `(feature_view|model):<contract name>:<feature name>`.
    """

    host: str

    model_alias: str
    model_name: str | None

    timeout: int = field(default=30)

    model_type: str = 'mlflow_server'

    @property
    def exposed_at_url(self) -> str | None:
        return self.host

    @property
    def as_markdown(self) -> str:
        return f"""Using a MLFlow server at `{self.host}`.
Assumes that it is the model: `{self.model_name}` with alias: `{self.model_alias}`.
This assums that the model signature are feature references on the aligned format.
Meaning each feature is on the following format `(feature_view|model):<contract name>:<feature name>`."""  # noqa: E501

    def get_model_version(self, model_name: str):
        from mlflow.tracking import MlflowClient

        mlflow_client = MlflowClient()
        return mlflow_client.get_model_version_by_alias(self.model_name or model_name, self.model_alias)

    def feature_refs(self) -> list[FeatureReference]:
        import json
        import mlflow

        info = mlflow.models.get_model_info(f"models:/{self.model_name}@{self.model_alias}")
        signature = info.signature_dict

        if not signature:
            return []

        def from_string(string: str) -> FeatureReference:
            splits = string.split(':')
            return FeatureReference(
                name=splits[-1],
                location=FeatureLocation.from_string(':'.join(splits[:-1])),
                # The dtype is meaningless in this context
                # sooooo should probably change this in the future
                dtype=FeatureType.string(),
            )

        return [from_string(feature['name']) for feature in json.loads(signature['inputs'])]

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return self.feature_refs()

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        features = await self.needed_features(store)
        req = store.store.requests_for_features(features)
        return req.request_result.entities

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        import polars as pl
        from httpx import AsyncClient
        from datetime import datetime, timezone

        pred_label = list(store.model.predictions_view.labels())[0]
        pred_at = store.model.predictions_view.event_timestamp
        model_version_column = store.model.predictions_view.model_version_column
        mv = None

        if model_version_column:
            mv = self.get_model_version(store.model.name)

        job = store.features_for(values)
        df = await job.to_polars()

        features = job.request_result.feature_columns

        async with AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.host}/invocations",
                json={'dataframe_records': df[features].to_dicts()},
            )
            response.raise_for_status()
            preds = response.json()['predictions']

        if pred_at:
            df = df.with_columns(
                pl.lit(datetime.now(timezone.utc)).alias(pred_at.name),
            )

        if mv and model_version_column:
            df = df.with_columns(
                pl.lit(mv.version).alias(model_version_column.name),
            )

        return df.with_columns(
            pl.Series(name=pred_label.name, values=preds),
        )
