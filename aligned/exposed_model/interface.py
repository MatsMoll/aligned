from __future__ import annotations

import polars as pl
from typing import TYPE_CHECKING
from dataclasses import dataclass
from aligned.retrival_job import RetrivalJob
from aligned.schemas.codable import Codable
from mashumaro.types import SerializableType
import logging

from aligned.schemas.feature import Feature, FeatureReference

if TYPE_CHECKING:
    from aligned.feature_store import ModelFeatureStore

logger = logging.getLogger(__name__)


class PredictorFactory:

    supported_predictors: dict[str, type[ExposedModel]]
    _shared: PredictorFactory | None = None

    def __init__(self):
        from aligned.exposed_model.mlflow import MLFlowServer, InMemMLFlowAlias
        from aligned.exposed_model.ollama import OllamaGeneratePredictor, OllamaEmbeddingPredictor

        self.supported_predictors = {}

        types: list[type[ExposedModel]] = [
            EnitityPredictor,
            OllamaGeneratePredictor,
            OllamaEmbeddingPredictor,
            MLFlowServer,
            InMemMLFlowAlias,
            ShadowModel,
            ABTestModel,
        ]
        for predictor in types:
            self.supported_predictors[predictor.model_type] = predictor

    @classmethod
    def shared(cls) -> PredictorFactory:
        if cls._shared:
            return cls._shared
        cls._shared = cls()
        return cls._shared


class ExposedModel(Codable, SerializableType):

    model_type: str

    @property
    def exposed_at_url(self) -> str | None:
        return None

    @property
    def as_markdown(self) -> str:
        raise NotImplementedError(type(self))

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        raise NotImplementedError(type(self))

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        raise NotImplementedError(type(self))

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        raise NotImplementedError(type(self))

    def _serialize(self) -> dict:
        assert (
            self.model_type in PredictorFactory.shared().supported_predictors
        ), f'Unknown predictor_type: {self.model_type}'
        return self.to_dict()

    def with_shadow(self, shadow_model: ExposedModel) -> ShadowModel:
        return ShadowModel(model=self, shadow_model=shadow_model)

    @classmethod
    def _deserialize(cls, value: dict) -> ExposedModel:
        name_type = value['model_type']
        if name_type not in PredictorFactory.shared().supported_predictors:
            raise ValueError(
                f"Unknown batch data source id: '{name_type}'.\nRemember to add the"
                ' data source to the BatchDataSourceFactory.supported_data_sources if'
                ' it is a custom type.'
            )
        del value['model_type']
        data_class = PredictorFactory.shared().supported_predictors[name_type]
        return data_class.from_dict(value)

    @staticmethod
    def ollama_generate(
        endpoint: str,
        model: str,
        prompt_template: str,
        input_features_versions: str,
    ) -> 'ExposedModel':
        from aligned.exposed_model.ollama import OllamaGeneratePredictor

        return OllamaGeneratePredictor(
            endpoint=endpoint,
            model_name=model,
            prompt_template=prompt_template,
            input_features_versions=input_features_versions,
        )

    @staticmethod
    def ollama_embedding(
        endpoint: str,
        model: str,
        input_features_versions: str,
        prompt_template: str,
        embedding_name: str | None = None,
    ) -> 'ExposedModel':
        from aligned.exposed_model.ollama import OllamaEmbeddingPredictor

        return OllamaEmbeddingPredictor(
            endpoint=endpoint,
            model_name=model,
            prompt_template=prompt_template,
            input_features_versions=input_features_versions,
            embedding_name=embedding_name or 'embedding',
        )

    @staticmethod
    def in_memory_mlflow(
        model_name: str,
        model_alias: str,
        model_contract_version_tag: str | None = None,
    ):
        from aligned.exposed_model.mlflow import in_memory_mlflow

        return in_memory_mlflow(
            model_name=model_name,
            model_alias=model_alias,
            model_contract_version_tag=model_contract_version_tag,
        )

    @staticmethod
    def mlflow_server(
        host: str,
        model_alias: str | None = None,
        model_name: str | None = None,
        model_contract_version_tag: str | None = None,
        timeout: int = 30,
    ):
        from aligned.exposed_model.mlflow import mlflow_server

        return mlflow_server(
            host=host,
            model_name=model_name,
            model_alias=model_alias,
            model_contract_version_tag=model_contract_version_tag,
            timeout=timeout,
        )


@dataclass
class EnitityPredictor(ExposedModel):

    endpoint: str

    input_features_versions: str

    model_type: str = 'entity'

    @property
    def exposed_at_url(self) -> str | None:
        return self.endpoint

    @property
    def as_markdown(self) -> str:
        return f"""Sending entities to as a JSON payload stored column wise: {self.endpoint}."""

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return store.feature_references_for(self.input_features_versions)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.using_version(self.input_features_versions).needed_entities()

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        from httpx import AsyncClient
        import polars as pl

        async with AsyncClient() as client:
            entities = (await values.to_polars()).to_dict(as_series=False)
            response = await client.post(self.endpoint, json=entities)
            response.raise_for_status()

        dict_data = dict(response.json())
        return pl.DataFrame(data=dict_data)


@dataclass
class ShadowModel(ExposedModel):

    model: ExposedModel
    shadow_model: ExposedModel

    model_type: str = 'shadow'

    @property
    def exposed_at_url(self) -> str | None:
        return self.model.exposed_at_url

    @property
    def as_markdown(self) -> str:
        return f"Main model: {self.model.as_markdown}.\n\nShadow model: {self.shadow_model.as_markdown}."

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        model_features = await self.model.needed_features(store)
        shadow_features = await self.shadow_model.needed_features(store)
        return model_features + shadow_features

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        model_entities = await self.model.needed_entities(store)
        shadow_entities = await self.shadow_model.needed_entities(store)
        return model_entities.union(shadow_entities)

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        pred_view = store.model.predictions_view

        model_df = await self.model.run_polars(values, store)
        shadow_df = await self.shadow_model.run_polars(values, store)

        if pred_view.is_shadow_model_flag:
            model_df = model_df.with_columns(pl.lit(False).alias(pred_view.is_shadow_model_flag.name))
            shadow_df = shadow_df.with_columns(pl.lit(True).alias(pred_view.is_shadow_model_flag.name))
        else:
            logger.info(
                'The model does not have a shadow model flag. '
                'This makes it harder to seperate shadow and production predictions.\n'
                'This can be set with `is_shadow = Bool().is_shadow_model_flag()`'
            )
        return model_df.vstack(shadow_df)


@dataclass
class ABTestModel(ExposedModel):
    """
    A model that runs multiple models in parallel and returns the
    result of one of them based on a weight and a random value.
    """

    models: list[tuple[ExposedModel, float]]

    model_type: str = 'abtest'

    @property
    def exposed_at_url(self) -> str | None:
        return self.models[0][0].exposed_at_url

    @property
    def as_markdown(self) -> str:
        model_definitions = [
            f"Model {i}: {model.as_markdown} with weight {weight}."
            for i, (model, weight) in enumerate(self.models)
        ]
        return '\n\n'.join(model_definitions)

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        features = []
        for model, _ in self.models:
            features += await model.needed_features(store)
        return features

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        entities = set()
        for model, _ in self.models:
            entities = entities.union(await model.needed_entities(store))
        return entities

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        import random

        total_weight = sum([weight for _, weight in self.models])
        total_sum = 0

        random_value = random.random()

        for model, weight in self.models:
            total_sum += weight / total_weight
            if random_value < total_sum:
                return await model.run_polars(values, store)

        return await self.models[-1][0].run_polars(values, store)


def ab_test_model(models: list[tuple[ExposedModel, float]]) -> ABTestModel:
    return ABTestModel(models=models)
