from __future__ import annotations

import polars as pl
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from aligned.retrival_job import RetrivalJob
from aligned.exposed_model.interface import ExposedModel

from aligned.schemas.feature import Feature, FeatureReference

if TYPE_CHECKING:
    from aligned.feature_store import ModelFeatureStore


def in_memory_mlflow(
    model_name: str,
    model_alias: str,
    model_contract_version_tag: str | None = None,
):
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
    model_contract_version_tag: str | None = None,
    timeout: int = 30,
):
    """Calls an MLFlow server to get the model predictions.

    This will load the model version from mlflow to get the expected model contract, and version name.
    """
    return MLFlowServer(
        host=host,
        model_name=model_name,
        model_alias=model_alias or 'champion',
        model_contract_version_tag=model_contract_version_tag,
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
                pl.lit(mv.run_id).alias(model_version_column.name),
            )

        return df.with_columns(
            pl.Series(name=pred_label.name, values=predictions),
        )


@dataclass
class MLFlowServer(ExposedModel):

    host: str

    model_alias: str
    model_name: str | None
    model_contract_version_tag: str | None

    timeout: int = field(default=30)

    model_type: str = 'mlflow_server'

    @property
    def exposed_at_url(self) -> str | None:
        return self.host

    @property
    def as_markdown(self) -> str:
        return f"""Using a MLFlow server at `{self.host}`.
Assumes that it is the model: `{self.model_name}` with alias: `{self.model_alias}`, and will load the features needed for that model based on the input version defined at tag `{self.model_contract_version_tag}`."""  # noqa: E501

    def get_model_version(self, model_name: str):
        from mlflow.tracking import MlflowClient

        mlflow_client = MlflowClient()
        return mlflow_client.get_model_version_by_alias(self.model_name or model_name, self.model_alias)

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
        mv = self.get_model_version(store.model.name)
        version = self.contract_version(mv)
        return store.feature_references_for(version)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        mv = self.get_model_version(store.model.name)
        version = self.contract_version(mv)
        return store.using_version(version).needed_entities()

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
                f'{self.host}/invocations', json={'dataframe_records': df[features].to_dicts()}
            )
            response.raise_for_status()
            preds = response.json()['predictions']

        if pred_at:
            df = df.with_columns(
                pl.lit(datetime.now(timezone.utc)).alias(pred_at.name),
            )

        if mv and model_version_column:
            df = df.with_columns(
                pl.lit(mv.run_id).alias(model_version_column.name),
            )

        return df.with_columns(
            pl.Series(name=pred_label.name, values=preds),
        )
