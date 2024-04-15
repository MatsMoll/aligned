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

        self.supported_predictors = {}

        types: list[type[ExposedModel]] = [
            EnitityPredictor,
            OllamaGeneratePredictor,
            OllamaEmbeddingPredictor,
            MLFlowServer,
            InMemMLFlowAlias,
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
    ) -> 'OllamaGeneratePredictor':

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
    ) -> 'OllamaEmbeddingPredictor':

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
class OllamaGeneratePredictor(ExposedModel):

    endpoint: str
    model_name: str

    prompt_template: str
    input_features_versions: str

    model_type: str = 'ollama_generate'

    @property
    def exposed_at_url(self) -> str | None:
        return self.endpoint

    def prompt_template_hash(self) -> str:
        from hashlib import sha256

        return sha256(self.prompt_template.encode(), usedforsecurity=False).hexdigest()

    @property
    def as_markdown(self) -> str:
        return f"""Sending a `generate` request to an Ollama server located at: {self.endpoint}.

This will use the model: `{self.model_name}` to generate the responses.

And use the prompt template:
```
{self.prompt_template}
```"""

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return store.feature_references_for(self.input_features_versions)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.using_version(self.input_features_versions).needed_entities()

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        from ollama import AsyncClient
        import polars as pl

        client = AsyncClient(host=self.endpoint)

        entities = await values.to_polars()

        features = store.feature_references_for(self.input_features_versions)
        expected_cols = {feat.name for feat in features}
        missing_cols = expected_cols - set(entities.columns)

        if missing_cols:
            entities = await (
                store.using_version(self.input_features_versions).features_for(values).to_polars()
            )

        prompts = entities

        ret_vals = []
        model_version = f"{self.prompt_template_hash()} -> {self.model_name}"

        for value in prompts.iter_rows(named=True):
            prompt = self.prompt_template.format(**value)

            response = await client.generate(self.model_name, prompt, stream=False)

            if isinstance(response, dict):
                response['model_version'] = model_version
            else:
                logger.info(f"Unable to log prompt to the Ollama response. Got: {type(response)}")

            ret_vals.append(response)

        return prompts.hstack(pl.DataFrame(ret_vals))


@dataclass
class OllamaEmbeddingPredictor(ExposedModel):

    endpoint: str
    model_name: str
    embedding_name: str

    prompt_template: str
    input_features_versions: str

    model_type: str = 'ollama_embedding'

    @property
    def exposed_at_url(self) -> str | None:
        return self.endpoint

    def prompt_template_hash(self) -> str:
        from hashlib import sha256

        return sha256(self.prompt_template.encode(), usedforsecurity=False).hexdigest()

    @property
    def as_markdown(self) -> str:
        return f"""Sending a `embedding` request to an Ollama server located at: {self.endpoint}.

This will use the model: `{self.model_name}` to generate the embeddings."""

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return store.feature_references_for(self.input_features_versions)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.using_version(self.input_features_versions).needed_entities()

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        from ollama import AsyncClient
        import polars as pl

        client = AsyncClient(host=self.endpoint)

        expected_cols = [feat.name for feat in store.feature_references_for(self.input_features_versions)]

        entities = await values.to_polars()
        missing_cols = set(expected_cols) - set(entities.columns)
        if missing_cols:
            entities = (
                await store.using_version(self.input_features_versions).features_for(values).to_polars()
            )

        prompts = entities

        ret_vals = []

        for value in prompts.iter_rows(named=True):
            prompt = self.prompt_template.format(**value)

            response = await client.embeddings(self.model_name, prompt)

            if isinstance(response, dict):
                embedding = response['embedding']  # type: ignore
            else:
                embedding = response

            ret_vals.append(embedding)

        model_version = f"{self.prompt_template_hash()} -> {self.model_name}"
        return prompts.hstack([pl.Series(name=self.embedding_name, values=ret_vals)]).with_columns(
            pl.lit(model_version).alias('model_version')
        )
