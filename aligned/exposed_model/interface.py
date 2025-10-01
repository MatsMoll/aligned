from __future__ import annotations

import polars as pl
from typing import TYPE_CHECKING, Any, AsyncIterable, Awaitable, Callable, Coroutine
from dataclasses import dataclass
from aligned.compiler.feature_factory import FeatureReferencable
from aligned.config_value import ConfigValue, LiteralValue
from aligned.retrieval_job import RetrievalJob
from aligned.schemas.codable import Codable
from mashumaro.types import SerializableType
import logging

from aligned.schemas.feature import (
    Feature,
    FeatureLocation,
    FeatureReference,
    FeatureType,
)

if TYPE_CHECKING:
    from aligned.feature_store import ContractStore
    from aligned.feature_store import ModelFeatureStore
    from aligned.schemas.model import Model
    from aligned.exposed_model.mlflow import MlflowConfig
    from aligned.exposed_model.openai import OpenAiConfig

logger = logging.getLogger(__name__)


class PredictorFactory:
    supported_predictors: dict[str, type[ExposedModel]]
    _shared: PredictorFactory | None = None

    def __init__(self) -> None:
        from aligned.exposed_model.mlflow import MLFlowServer, InMemMLFlowAlias
        from aligned.exposed_model.ollama import (
            OllamaGeneratePredictor,
            OllamaEmbeddingPredictor,
        )
        from aligned.exposed_model.sentence_transformer import (
            SentenceTransformerPredictor,
        )
        from aligned.exposed_model.langchain import LangChain
        from aligned.exposed_model.openai import (
            OpenAiEmbeddingPredictor,
            OpenAiPromptModel,
            OpenAiExtractModel,
        )
        from aligned.exposed_model.partitioned import PartitionedModel
        from aligned.exposed_model.multiple import MultipleModels

        self.supported_predictors = {}

        types: list[type[ExposedModel]] = [
            EnitityPredictor,
            OllamaGeneratePredictor,
            OllamaEmbeddingPredictor,
            MLFlowServer,
            InMemMLFlowAlias,
            ShadowModel,
            ABTestModel,
            DillPredictor,
            SentenceTransformerPredictor,
            LangChain,
            PartitionedModel,
            OpenAiEmbeddingPredictor,
            OpenAiPromptModel,
            OpenAiExtractModel,
            MultipleModels,
            PolarsExpression,
            CodePredictor,
        ]
        for predictor in types:
            self.supported_predictors[predictor.model_type] = predictor

    @classmethod
    def shared(cls) -> PredictorFactory:
        if cls._shared:
            return cls._shared
        cls._shared = cls()
        return cls._shared


class PromptModel:
    @property
    def precomputed_prompt_key(self) -> str | None:
        """
        This is the property that contains the fully compiled prompt.
        Meaning a user can bypass the prompt templating step.

        This is useful in some scanarios where we want to do similarity search
        when the prompt components do not make sense to provide.
        """
        return None


class VersionedModel:
    async def model_version(self) -> str:
        raise NotImplementedError(type(self))


class ExposedModel(Codable, SerializableType):
    model_type: str

    @property
    def exposed_at_url(self) -> str | None:
        return None

    @property
    def as_markdown(self) -> str:
        raise NotImplementedError(type(self))

    def needed_configs(self) -> list[ConfigValue]:
        "Returns the config variables to set in order to use the model."
        return []

    async def depends_on(self) -> list[FeatureLocation]:
        """
        The data artefacts that the model depends on. Which is not the input features.
        This is useful for e.g. RAG systems, as we can describe which documents a model depends on
        Or something like a vector database that we assume to be up to date.
        """
        return []

    def with_contract(self, model: Model) -> ExposedModel:
        return self

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        raise NotImplementedError(type(self))

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        raise NotImplementedError(type(self))

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        raise NotImplementedError(type(self))

    async def potential_drift_from_model(self, old_model: ExposedModel) -> str | None:
        raise NotImplementedError(type(self))

    def _serialize(self) -> dict:
        assert (
            self.model_type in PredictorFactory.shared().supported_predictors
        ), f"Unknown predictor_type: {self.model_type}"
        return self.to_dict()

    def with_shadow(self, shadow_model: ExposedModel) -> ShadowModel:
        return ShadowModel(model=self, shadow_model=shadow_model)

    @classmethod
    def _deserialize(cls, value: dict) -> ExposedModel:
        name_type = value["model_type"]
        if name_type not in PredictorFactory.shared().supported_predictors:
            raise ValueError(
                f"Unknown batch data source id: '{name_type}'.\nRemember to add the"
                " data source to the BatchDataSourceFactory.supported_data_sources if"
                " it is a custom type."
            )
        del value["model_type"]
        data_class = PredictorFactory.shared().supported_predictors[name_type]
        return data_class.from_dict(value)

    @staticmethod
    def polars_predictor(
        callable: Callable[
            [pl.DataFrame, ModelFeatureStore], Coroutine[None, None, pl.DataFrame]
        ],
        features: list[FeatureReferencable] | None = None,
    ) -> "ExposedModel":
        refs = [feat.feature_reference() for feat in features or []]
        return CodePredictor.from_function(callable, refs)  # type: ignore

    @staticmethod
    def ollama_generate(
        endpoint: str,
        model: str,
        prompt_template: str,
        input_features_versions: str,
    ) -> "ExposedModel":
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
        precomputed_prompt_key: str = "full_prompt",
    ) -> "ExposedModel":
        from aligned.exposed_model.ollama import OllamaEmbeddingPredictor

        return OllamaEmbeddingPredictor(
            endpoint=endpoint,
            model_name=model,
            prompt_template=prompt_template,
            input_features_versions=input_features_versions,
            embedding_name=embedding_name or "embedding",
            precomputed_prompt_key_overwrite=precomputed_prompt_key,
        )

    @staticmethod
    def in_memory_mlflow(
        model_name: str,
        model_alias: str = "champion",
        reference_tag: str = "feature_refs",
        mlflow_config: MlflowConfig | None = None,
    ) -> "ExposedModel":
        from aligned.exposed_model.mlflow import in_memory_mlflow

        return in_memory_mlflow(
            model_name=model_name,
            model_alias=model_alias,
            reference_tag=reference_tag,
            mlflow_config=mlflow_config,
        )

    @staticmethod
    def mlflow_server(
        host: str,
        model_alias: str = "champion",
        model_name: str | None = None,
        timeout: int = 30,
    ) -> "ExposedModel":
        from aligned.exposed_model.mlflow import mlflow_server

        return mlflow_server(
            host=host,
            model_name=model_name,
            model_alias=model_alias,
            timeout=timeout,
        )


class StreamablePredictor:
    async def stream_predict(
        self, input: dict[str, Any]
    ) -> AsyncIterable[dict[str, Any]]:
        raise NotImplementedError(type(self))


@dataclass
class DillPredictor(ExposedModel, VersionedModel):
    function: bytes
    function_id: str
    feature_refs: list[FeatureReference] | None = None

    model_type: str = "dill_predictor"

    @property
    def exposed_at_url(self) -> str | None:
        return None

    @property
    def as_markdown(self) -> str:
        return "A function stored in a dill file."

    async def model_version(self) -> str:
        return self.function_id

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        default = store.model.features.default_version
        return store.feature_references_for(store.selected_version or default)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.request().request_result.entities

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        import dill
        import inspect

        function = dill.loads(self.function)

        if self.feature_refs:
            features = await store.store.features_for(
                values, self.feature_refs
            ).to_polars()
        else:
            features = await store.input_features_for(values).to_polars()

        if inspect.iscoroutinefunction(function):
            return await function(features, store)
        else:
            return function(features, store)


@dataclass
class CodePredictor(ExposedModel, VersionedModel):
    code: str
    function_name: str
    feature_refs: list[FeatureReference] | None = None

    model_type: str = "code_pred"

    @property
    def exposed_at_url(self) -> str | None:
        return None

    @property
    def as_markdown(self) -> str:
        return f"Predicts using the following function: \n```python\n{self.code}\n```"

    async def model_version(self) -> str:
        from hashlib import sha256

        return sha256(self.code.encode()).hexdigest()

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        default = store.model.features.default_version
        return store.feature_references_for(store.selected_version or default)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.request().request_result.entities

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        import inspect

        if self.function_name not in locals():
            exec(self.code)

        function = locals()[self.function_name]
        args = inspect.signature(function).parameters

        if len(args) == 1:
            if self.feature_refs:
                features = await store.store.features_for(
                    values, self.feature_refs
                ).to_polars()
            else:
                features = await store.input_features_for(values).to_polars()

            if inspect.iscoroutinefunction(function):
                pred = await function(features)
            else:
                pred = function(features)
        else:
            features = await values.to_polars()
            if inspect.iscoroutinefunction(function):
                pred = await function(features, store.store)
            else:
                pred = function(features, store.store)

        if isinstance(pred, pl.DataFrame):
            return pred

        assert isinstance(pred, pl.Series), f"Expected a Series but got {type(pred)}"

        pred_view = store.model.predictions_view
        pred_columns = pred_view.labels()
        if len(pred_columns) != 1:
            raise ValueError(
                f"Expected exactly one prediction column, got {len(pred_columns)} columns."
            )

        result = features.with_columns(pred.alias(next(iter(pred_columns)).name))

        if pred_view.model_version_column:
            version = await self.model_version()
            result = result.with_columns(
                pl.lit(version).alias(pred_view.model_version_column.name)
            )

        return result

    @staticmethod
    def from_function(
        function: Callable[[pl.DataFrame], pl.Series]
        | Callable[[pl.DataFrame], Awaitable[pl.Series]]
        | Callable[[pl.DataFrame, ContractStore], pl.Series]
        | Callable[[pl.DataFrame, ContractStore], Awaitable[pl.Series]],
        feature_refs: list[FeatureReference] | None = None,
    ) -> CodePredictor:
        import dill
        import inspect
        from uuid import uuid4

        raw_code = inspect.getsource(function)

        if function.__name__ == "<lambda>":
            lambda_body = raw_code.split("lambda ")[1]
            lambda_name = f"a{uuid4()}".replace("-", "_")
            args, body = lambda_body.split(":")

            new_body = body.rstrip("), \n")

            code = f"def {lambda_name}({args}):\n    return {new_body}"

            return CodePredictor(code, lambda_name)

        function_name = dill.source.getname(function)
        assert isinstance(function_name, str), "Need a function name"

        code = ""

        indents: int | None = None
        start_signature = f"def {function_name}"

        if inspect.iscoroutinefunction(function):
            start_signature = f"async {start_signature}"

        for line in raw_code.splitlines(keepends=True):
            if indents:
                if len(line) > indents:
                    code += line[:indents].lstrip() + line[indents:]
                else:
                    code += line
            elif start_signature in line:
                stripped = line.lstrip()
                indents = len(line) - len(stripped)
                stripped = stripped.replace(
                    f"{start_signature}(self,", f"{start_signature}("
                )
                code += stripped
            else:
                code += line

        return CodePredictor(code, function_name, feature_refs=feature_refs)


@dataclass
class PolarsExpression(ExposedModel, VersionedModel):
    expression: str
    should_filter_nulls: bool = True
    model_type: str = "polars_expression"

    @property
    def exposed_at_url(self) -> str | None:
        return None

    @property
    def as_markdown(self) -> str:
        return f"A function stored in a {self.expression}."

    async def model_version(self) -> str:
        from hashlib import sha256

        return sha256(self.expression.encode()).hexdigest()

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        default = store.model.features.default_version
        return store.feature_references_for(store.selected_version or default)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.request().request_result.entities

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        pred_view = store.model.predictions_view
        labels = pred_view.labels()

        assert len(labels) == 1, f"Expected only one label got {len(labels)}"

        inputs = await store.input_features_for(values).to_polars()

        name = next(iter(labels)).name

        preds = inputs.with_columns(
            pl.Expr.deserialize(self.expression.encode(), format="json").alias(name)
        )
        if self.should_filter_nulls:
            preds = preds.filter(pl.col(name).is_not_null())

        return preds


@dataclass
class EnitityPredictor(ExposedModel):
    endpoint: str

    input_features_versions: str

    model_type: str = "entity"

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

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
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

    model_type: str = "shadow"

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

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        pred_view = store.model.predictions_view

        model_df = await self.model.run_polars(values, store)
        shadow_df = await self.shadow_model.run_polars(values, store)

        if pred_view.is_shadow_model_flag:
            model_df = model_df.with_columns(
                pl.lit(False).alias(pred_view.is_shadow_model_flag.name)
            )
            shadow_df = shadow_df.with_columns(
                pl.lit(True).alias(pred_view.is_shadow_model_flag.name)
            )
        else:
            logger.info(
                "The model does not have a shadow model flag. "
                "This makes it harder to separate shadow and production predictions.\n"
                "This can be set with `is_shadow = Bool().is_shadow_model_flag()`"
            )
        return model_df.vstack(shadow_df)


@dataclass
class ABTestModel(ExposedModel):
    """
    A model that runs multiple models in parallel and returns the
    result of one of them based on a weight and a random value.
    """

    models: list[tuple[ExposedModel, float]]

    model_type: str = "abtest"

    @property
    def exposed_at_url(self) -> str | None:
        return self.models[0][0].exposed_at_url

    @property
    def as_markdown(self) -> str:
        model_definitions = [
            f"Model {i}: {model.as_markdown} with weight {weight}."
            for i, (model, weight) in enumerate(self.models)
        ]
        return "\n\n".join(model_definitions)

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        features = []
        for model, _ in self.models:
            features += await model.needed_features(store)
        return features

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        entities: set[Feature] = set()
        for model, _ in self.models:
            entities = entities.union(await model.needed_entities(store))
        return entities

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        import random

        total_weight = sum([weight for _, weight in self.models])
        total_sum: float = 0.0

        random_value = random.random()

        for model, weight in self.models:
            total_sum += weight / total_weight
            if random_value < total_sum:
                return await model.run_polars(values, store)

        return await self.models[-1][0].run_polars(values, store)


def ab_test_model(models: list[tuple[ExposedModel, float]]) -> ABTestModel:
    return ABTestModel(models=models)


@dataclass
class DillFunction(ExposedModel, VersionedModel):
    function: bytes

    model_type: str = "dill_function"

    @property
    def exposed_at_url(self) -> str | None:
        return None

    @property
    def as_markdown(self) -> str:
        return "A function stored in a dill file."

    async def model_version(self) -> str:
        from hashlib import md5

        return md5(self.function, usedforsecurity=False).hexdigest()

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        default = store.model.features.default_version
        return store.feature_references_for(store.selected_version or default)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.request().request_result.entities

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        import dill
        import inspect

        function = dill.loads(self.function)
        if inspect.iscoroutinefunction(function):
            return await function(values, store)
        else:
            return function(values, store)


def python_function(function: Callable[[pl.DataFrame], pl.Series]) -> ExposedModel:
    return CodePredictor.from_function(function)


def openai_embedding(
    model: str,
    config: OpenAiConfig | None = None,
    batch_on_n_chunks: int | None = 100,
    prompt_template: str | None = None,
) -> ExposedModel:
    """
    Returns an OpenAI embedding model.

    ```python
    @model_contract(
        input_features=[MyFeature().name],
        exposed_model=openai_embedding("text-embedding-3-small"),
    )
    class MyEmbedding:
        my_entity = Int32().as_entity()
        name = String()
        embedding = Embedding(1536)
        predicted_at = EventTimestamp()

    embeddings = await store.model(MyEmbedding).predict_over({
        "my_entity": [1, 2, 3],
        "name": ["Hello", "World", "foo"]
    }).to_polars()
    ```


    Args:
        model (str): the model to use. Look at the OpenAi docs to find the correct one.
        batch_on_n_chunks (int): When to change to the batch API. Given that the batch size is too big.
        prompt_template (str): A custom prompt template if wanted. The default will be based on the input features.

    Returns:
        ExposedModel: a model that sends embedding requests to OpenAI

    """
    from aligned.exposed_model.openai import OpenAiEmbeddingPredictor, OpenAiConfig

    return OpenAiEmbeddingPredictor(
        model=model,
        config=config or OpenAiConfig(),
        batch_on_n_chunks=batch_on_n_chunks,
        prompt_template=prompt_template or "",
    )


def openai_completion(
    model: str, prompt_template: str | None = None, config: OpenAiConfig | None = None
) -> ExposedModel:
    """
    Returns an OpenAI completion model.

    ```python
    @model_contract(
        input_features=[MyFeature().name],
        exposed_model=openai_completion(""),
    )
    class MyCompletion:
        my_entity = Int32().as_entity()
        name = String()
        response = String().as_prompt_completion()
        predicted_at = EventTimestamp()

    embeddings = await store.model(MyCompletion).predict_over({
        "my_entity": [1, 2, 3],
        "name": ["Hello", "World", "foo"]
    }).to_polars()
    ```


    Args:
        model (str): the model to use. Look at the OpenAi docs to find the correct one.
        batch_on_n_chunks (int): When to change to the batch API. Given that the batch size is too big.
        prompt_template (str): A custom prompt template if wanted. The default will be based on the input features.

    Returns:
        ExposedModel: a model that sends embedding requests to OpenAI
    """

    from aligned.exposed_model.openai import OpenAiPromptModel, OpenAiConfig

    return OpenAiPromptModel(
        model=model,
        prompt_template=prompt_template or "",
        config=config or OpenAiConfig(),
    )


def openai_extraction(
    model: str,
    extraction_description: str | None = None,
    config: OpenAiConfig | None = None,
) -> ExposedModel:
    from aligned.exposed_model.openai import OpenAiExtractModel, OpenAiConfig

    return OpenAiExtractModel(
        model=model,
        extract_task_description=extraction_description
        or OpenAiExtractModel.default_extract_message(),
        config=config or OpenAiConfig(),
    )


def ollama_extraction(
    model: str,
    base_url: str | ConfigValue = "http://localhost:11434/v1",
    api_key: str | ConfigValue = "ollama",
    extraction_description: str | None = None,
) -> ExposedModel:
    from aligned.exposed_model.interface import openai_extraction
    from aligned.exposed_model.openai import OpenAiConfig

    return openai_extraction(
        model,
        extraction_description=extraction_description,
        config=OpenAiConfig(
            api_key=LiteralValue.from_value(api_key),
            base_url=LiteralValue.from_value(base_url),
        ),
    )


def partitioned_on(
    key: str, partitions: dict[str, ExposedModel], default_partition: str | None = None
) -> ExposedModel:
    """Returns an model that routes the inference request to a new model based on a partition key

    ```python
    @model_contract(
        input_features=[MyFeature().name],
        exposed_model=partitioned_on(
            "lang",
            partitions={
                "no": openai_embedding("text-embedding-3-large"),
                "en": openai_embedding("text-embedding-ada-002"),
            },
            default_partition="no"
        ),
    )
    class MyEmbedding:
        my_entity = Int32().as_entity()
        name = String()
        lang = String()
        embedding = Embedding(1536)
        predicted_at = EventTimestamp()

    embeddings = await store.model(MyEmbedding).predict_over({
        "my_entity": [1, 2, 3],
        "name": ["Hello", "Hei", "Hola"],
        "lang": ["en", "no", "es"]
    }).to_polars()
    ```
    """
    from aligned.exposed_model.partitioned import PartitionedModel

    return PartitionedModel(
        Feature(key, FeatureType.string()),
        partitions=partitions,
        default_partition=default_partition,
    )


def multiple_models(*models: ExposedModel) -> ExposedModel:
    from aligned.exposed_model.multiple import MultipleModels

    return MultipleModels(list(models))


def polars_predictor(
    callable: Callable[
        [pl.DataFrame, ModelFeatureStore], Coroutine[None, None, pl.DataFrame]
    ],
    features: list[FeatureReferencable] | None = None,
) -> ExposedModel:
    return ExposedModel.polars_predictor(callable, features)


def polars_expression(expr: pl.Expr, should_filter_nulls: bool = True) -> ExposedModel:
    return PolarsExpression(
        expr.meta.serialize(format="json"), should_filter_nulls=should_filter_nulls
    )
