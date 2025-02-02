from __future__ import annotations

from mlflow.exceptions import MlflowException
import polars as pl
from typing import TYPE_CHECKING, Any, Iterable
from dataclasses import dataclass, field
from aligned.config_value import ConfigValue, EnvironmentValue, LiteralValue
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import RetrivalJob
from aligned.exposed_model.interface import ExposedModel

from aligned.lazy_imports import mlflow

from aligned.schemas.feature import Feature, FeatureReference

if TYPE_CHECKING:
    from aligned.feature_store import ModelFeatureStore
    from mlflow.pyfunc import PyFuncModel

try:
    from mlflow.models import ModelSignature
    from mlflow.types.schema import Schema, ColSpec, TensorSpec

    def mlflow_spec(feature: Feature) -> ColSpec | TensorSpec | str:
        dtype = feature.dtype

        if dtype.name == 'float':
            return ColSpec('float', name=feature.name)
        elif dtype.name == 'double':
            return ColSpec('double', name=feature.name)
        elif dtype.name == 'string':
            return ColSpec('string', name=feature.name)
        elif dtype.is_numeric:
            return ColSpec('integer', name=feature.name)
        elif dtype.is_datetime:
            return ColSpec('datetime', name=feature.name)
        elif dtype.is_embedding:
            import numpy as np

            return TensorSpec(
                type=np.dtype(np.float32),
                shape=(-1, dtype.embedding_size()),
                name=feature.name,
            )
        return dtype.name

    def signature_for_features(inputs: Iterable[Feature], outputs: Iterable[Feature]) -> ModelSignature:

        output_schema: Schema | None = None
        if outputs:
            output_schema = Schema([mlflow_spec(label) for label in outputs])  # type: ignore

        all_features = []
        for feature in inputs:
            all_features.append(mlflow_spec(feature))

        return ModelSignature(inputs=Schema(all_features), outputs=output_schema)

    def signature_for_contract(model_contract: ModelFeatureStore) -> ModelSignature:
        input_req = model_contract.request()
        input_reqs = input_req.needed_requests
        pred_view = model_contract.model.predictions_view

        all_features = []
        for request in input_reqs:
            for feature in request.returned_features:
                if feature.name not in input_req.features_to_include:
                    continue
                all_features.append(feature)

        return signature_for_features(inputs=all_features, outputs=pred_view.labels())

except ModuleNotFoundError:
    pass


def reference_metadata_for_features(features: list[FeatureReference]) -> list[str]:
    return [feat.identifier for feat in sorted(features, key=lambda feat: feat.name)]


def reference_metadata_for_input(requests: list[RetrivalRequest] | RetrivalRequest) -> list[str]:
    if isinstance(requests, RetrivalRequest):
        requests = [requests]

    all_features: list[FeatureReference] = []
    for req in requests:
        all_features.extend(feat.as_reference(req.location) for feat in req.returned_features)
    return reference_metadata_for_features(all_features)


def references_from_metadata(
    metadata: dict[str, Any], reference_key: str = 'feature_refs'
) -> list[FeatureReference] | None:
    """
    Decodes the feature references in stored in a metadata key.

    ```python
    mlflow.
    ```
    """

    if reference_key not in metadata:
        return None

    refs = metadata[reference_key]
    assert isinstance(refs, list), f"Expected a list got '{type(refs)}'"

    decoded_refs = [FeatureReference.from_string(ref) for ref in refs]
    return [ref for ref in decoded_refs if ref is not None]


@dataclass
class MlflowConfig:

    tracking_uri: ConfigValue = field(default_factory=lambda: EnvironmentValue('MLFLOW_TRACKING_URI'))
    registry_uri: ConfigValue = field(default_factory=lambda: EnvironmentValue('MLFLOW_REGISTRY_URI'))

    def configs(self) -> list[ConfigValue]:
        return [self.tracking_uri, self.registry_uri]

    def client(self) -> mlflow.MlflowClient:
        mlflow.set_registry_uri(self.registry_uri.read())

        return mlflow.MlflowClient(
            tracking_uri=self.tracking_uri.read(), registry_uri=self.registry_uri.read()
        )

    def __enter__(self) -> mlflow.MlflowClient:
        self._old_tracking_uri = mlflow.get_tracking_uri()
        self._old_registry_uri = mlflow.get_registry_uri()

        client = self.client()

        mlflow.set_tracking_uri(client.tracking_uri)
        mlflow.set_registry_uri(client._registry_uri)

        return client

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old_tracking_uri:
            mlflow.set_tracking_uri(self._old_tracking_uri)
            self._old_tracking_uri = None

        if self._old_registry_uri:
            mlflow.set_registry_uri(self._old_registry_uri)
            self._old_registry_uri = None

    @staticmethod
    def databricks_unity_catalog() -> MlflowConfig:
        return MlflowConfig(
            tracking_uri=LiteralValue('databricks'),
            registry_uri=LiteralValue('databricks-uc'),
        )


def in_memory_mlflow(
    model_name: str,
    model_alias: str = 'champion',
    reference_tag: str = 'feature_refs',
    mlflow_config: MlflowConfig | None = None,
) -> ExposedModel:
    """A model that is loaded from MLFlow using the given model name and alias.

    This will also run in memory, and not require that `mlflow` is installed.
    """
    return InMemMLFlowAlias(
        model_name=model_name,
        model_alias=model_alias,
        reference_tag=reference_tag,
        mlflow_config=mlflow_config or MlflowConfig(),
    )


def mlflow_server(
    host: str,
    model_alias: str = 'champion',
    model_name: str | None = None,
    timeout: int = 30,
    mlflow_config: MlflowConfig | None = None,
) -> ExposedModel:
    """Calls an MLFlow server to get the model predictions.

    This will load the features described in the model signature, and pass them to the server.
    """
    return MLFlowServer(
        host=host,
        model_name=model_name,
        model_alias=model_alias,
        timeout=timeout,
        mlflow_config=mlflow_config or MlflowConfig(),
    )


@dataclass
class InMemMLFlowAlias(ExposedModel):

    model_name: str
    model_alias: str

    mlflow_config: MlflowConfig

    drop_invalid_rows: bool = True
    reference_tag: str = 'feature_refs'
    model_type: str = 'latest_mlflow'

    @property
    def exposed_at_url(self) -> str | None:
        return None

    @property
    def as_markdown(self) -> str:
        return f"""Using the latest MLFlow model: `{self.model_name}`."""

    def needed_configs(self) -> list[ConfigValue]:
        return self.mlflow_config.configs()

    def get_model_version(self, client: mlflow.MlflowClient | None = None):
        if client is None:
            client = self.mlflow_config.client()
        return client.get_model_version_by_alias(self.model_name, self.model_alias)

    def feature_refs(
        self, client: mlflow.MlflowClient, store: ModelFeatureStore, model: PyFuncModel | None = None
    ) -> list[FeatureReference]:

        if model is not None and model.metadata.metadata is not None:
            refs = references_from_metadata(
                metadata=model.metadata.metadata, reference_key=self.reference_tag
            )
            if refs is not None:
                return refs

        version = client.get_model_version_by_alias(self.model_name, self.model_alias)
        refs = references_from_metadata(version.tags, self.reference_tag)
        if refs:
            return refs

        return store.input_features()

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return self.feature_refs(self.mlflow_config.client(), store, None)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        refs = await self.needed_features(store)
        return store.store.requests_for_features(refs).entities()

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        import polars as pl
        import pandas as pd
        from datetime import datetime, timezone

        pred_label = list(store.model.predictions_view.labels())[0]
        pred_at = store.model.predictions_view.event_timestamp
        model_version_column = store.model.predictions_view.model_version_column
        mv = None

        with self.mlflow_config as client:
            if model_version_column:
                mv = self.get_model_version(client)

            model_uri = f"models:/{self.model_name}@{self.model_alias}"
            model = mlflow.pyfunc.load_model(model_uri)
            feature_refs = self.feature_refs(client, store, model)

        job = store.store.features_for(values, feature_refs)
        if self.drop_invalid_rows:
            job = job.drop_invalid()
        df = await job.to_polars()

        features = [feat.name for feat in feature_refs]
        try:
            predictions = model.predict(df[features])
        except MlflowException:
            predictions = model.predict(df[features].to_pandas())
            if not isinstance(predictions, pd.Series):
                predictions = pd.Series(predictions)
            predictions = pl.from_pandas(predictions, include_index=False)

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

    mlflow_config: MlflowConfig
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
        mlflow_client = self.mlflow_config.client()
        return mlflow_client.get_model_version_by_alias(self.model_name or model_name, self.model_alias)

    def feature_refs(self) -> list[FeatureReference]:
        import json

        with self.mlflow_config as _:
            info = mlflow.models.get_model_info(f"models:/{self.model_name}@{self.model_alias}")
            signature = info.signature_dict

        if not signature:
            return []

        refs = [FeatureReference.from_string(feature['name']) for feature in json.loads(signature['inputs'])]

        return [ref for ref in refs if ref]

    def needed_configs(self) -> list[ConfigValue]:
        return self.mlflow_config.configs()

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
