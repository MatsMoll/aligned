from dataclasses import dataclass, field
from datetime import datetime, timezone
from aligned.compiler.model import ModelContractWrapper
from aligned.compiler.feature_factory import (
    CouldBeEntityFeature,
    FeatureFactory,
    FeatureReferencable,
    Int32,
    String,
    List,
    Int64,
    EventTimestamp,
)
import logging

from aligned.data_source.batch_data_source import CodableBatchDataSource
from aligned.exposed_model.interface import ExposedModel, PromptModel
from aligned.schemas.feature import Feature, FeatureReference
from aligned.retrieval_job import RetrievalJob
import polars as pl
from aligned.feature_store import ModelFeatureStore
from aligned.schemas.model import Model


logger = logging.getLogger(__name__)


@dataclass
class OllamaGeneratePredictor(ExposedModel):
    endpoint: str
    model_name: str

    prompt_template: str
    input_features_versions: str

    model_type: str = "ollama_generate"

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

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        from ollama import AsyncClient
        import polars as pl

        client = AsyncClient(host=self.endpoint)

        entities = await values.to_polars()

        features = store.feature_references_for(self.input_features_versions)
        expected_cols = {feat.name for feat in features}
        missing_cols = expected_cols - set(entities.columns)

        if missing_cols:
            entities = await (
                store.using_version(self.input_features_versions)
                .features_for(values)
                .to_polars()
            )

        prompts = entities

        ret_vals = []
        model_version = f"{self.prompt_template_hash()} -> {self.model_name}"

        for value in prompts.iter_rows(named=True):
            prompt = self.prompt_template.format(**value)

            response = await client.generate(self.model_name, prompt, stream=False)

            if isinstance(response, dict):
                response["model_version"] = model_version
            else:
                logger.info(
                    f"Unable to log prompt to the Ollama response. Got: {type(response)}"
                )

            ret_vals.append(response)

        return prompts.hstack(pl.DataFrame(ret_vals))


@dataclass
class OllamaEmbeddingPredictorWithRef(ExposedModel, PromptModel):
    endpoint: str
    model_name: str

    embedding_name: str = field(default="")
    feature_references: list[FeatureReference] = field(default_factory=list)
    prompt_template: str = field(default="")

    precomputed_prompt_key_overwrite: str = "full_prompt"
    model_type: str = "ollama_embedding"

    @property
    def precomputed_prompt_key(self) -> str | None:
        return self.precomputed_prompt_key_overwrite

    @property
    def exposed_at_url(self) -> str | None:
        return self.endpoint

    def prompt_template_hash(self) -> str:
        from hashlib import sha256

        return sha256(self.prompt_template.encode(), usedforsecurity=False).hexdigest()

    def with_contract(self, model: Model) -> ExposedModel:
        if len(model.features.versions) != 1:
            assert self.feature_references != []
            return self

        if self.prompt_template == "":
            refs = model.feature_references()
            if len(refs) == 1:
                self.prompt_template += f"{{{list(refs)[0].name}}}"
            else:
                for feature in refs:
                    self.prompt_template += f"{feature.name}: {{{feature.name}}}"

        if self.embedding_name == "":
            embeddings = model.predictions_view.embeddings()
            assert len(embeddings) == 1
            self.embedding_name = embeddings[0].name

        return self

    async def potential_drift_from_model(self, old_model: ExposedModel) -> str | None:
        """
        Checks if a change in model can lead to a potential distribution shift.

        Returns:
            str: A message explaining the potential drift.
        """
        if not isinstance(old_model, OllamaEmbeddingPredictor):
            return None

        changes = ""
        if old_model.model_name != self.model_name:
            changes += f"Model name changed from `{old_model.model_name}` to `{self.model_name}`.\n"

        if old_model.prompt_template != self.prompt_template:
            changes += f"Prompt template changed from `{old_model.prompt_template}` to `{self.prompt_template}`.\n"

        if changes:
            return changes
        else:
            return None

    @property
    def as_markdown(self) -> str:
        return f"""Sending a `embedding` request to an Ollama server located at: {self.endpoint}.

This will use the model: `{self.model_name}` to generate the embeddings."""

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return self.feature_references

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.store.requests_for_features(
            self.feature_references
        ).request_result.entities

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        from ollama import AsyncClient
        import polars as pl

        client = AsyncClient(host=self.endpoint)

        expected_cols = [feat.name for feat in self.feature_references]
        entities = await values.to_polars()

        prompts = []

        if self.precomputed_prompt_key_overwrite in entities.columns:
            prompts = entities[self.precomputed_prompt_key_overwrite].to_list()
        else:
            missing_cols = set(expected_cols) - set(entities.columns)
            if missing_cols:
                entities = await store.store.features_for(
                    values, features=self.feature_references
                ).to_polars()

            for index, value in enumerate(entities.iter_rows(named=True)):
                logger.info(f"Processing row {index + 1}/{len(prompts)}")

                prompt = self.prompt_template.format(**value)
                prompts.append(prompt)

            entities = entities.with_columns(
                pl.Series(name=self.precomputed_prompt_key_overwrite, values=prompts)
            )

        ret_vals = []

        for prompt in prompts:
            response = await client.embeddings(self.model_name, prompt)

            if isinstance(response, dict):
                embedding = response["embedding"]  # type: ignore
            else:
                embedding = response

            ret_vals.append(embedding)

        pred_view = store.model.predictions_view
        if pred_view.model_version_column:
            model_version = f"{self.prompt_template_hash()} -> {self.model_name}"
            model_version_name = pred_view.model_version_column.name
            entities = entities.with_columns(
                pl.lit(model_version).alias(model_version_name)
            )

        if pred_view.event_timestamp:
            new_et = pred_view.event_timestamp.name
            existing_et = values.request_result.event_timestamp
            need_to_add_et = new_et not in entities.columns

            if existing_et and need_to_add_et and existing_et in entities.columns:
                logger.info(
                    f"Using existing event timestamp `{existing_et}` as new timestamp."
                )
                entities = entities.with_columns(pl.col(existing_et).alias(new_et))
            elif need_to_add_et:
                logger.info("No event timestamp using now as the timestamp.")
                entities = entities.with_columns(
                    pl.lit(datetime.now(tz=timezone.utc)).alias(
                        pred_view.event_timestamp.name
                    )
                )

        return entities.hstack([pl.Series(name=self.embedding_name, values=ret_vals)])


@dataclass
class OllamaEmbeddingPredictor(ExposedModel, PromptModel):
    endpoint: str
    model_name: str
    embedding_name: str

    prompt_template: str
    input_features_versions: str

    precomputed_prompt_key_overwrite: str
    model_type: str = "ollama_embedding"

    @property
    def precomputed_prompt_key(self) -> str | None:
        return self.precomputed_prompt_key_overwrite

    @property
    def exposed_at_url(self) -> str | None:
        return self.endpoint

    def prompt_template_hash(self) -> str:
        from hashlib import sha256

        return sha256(self.prompt_template.encode(), usedforsecurity=False).hexdigest()

    async def potential_drift_from_model(self, old_model: ExposedModel) -> str | None:
        """
        Checks if a change in model can lead to a potential distribution shift.

        Returns:
            str: A message explaining the potential drift.
        """
        if not isinstance(old_model, OllamaEmbeddingPredictor):
            return None

        changes = ""
        if old_model.model_name != self.model_name:
            changes += f"Model name changed from `{old_model.model_name}` to `{self.model_name}`.\n"

        if old_model.prompt_template != self.prompt_template:
            changes += f"Prompt template changed from `{old_model.prompt_template}` to `{self.prompt_template}`.\n"

        if changes:
            return changes
        else:
            return None

    @property
    def as_markdown(self) -> str:
        return f"""Sending a `embedding` request to an Ollama server located at: {self.endpoint}.

This will use the model: `{self.model_name}` to generate the embeddings."""

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return store.feature_references_for(self.input_features_versions)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.using_version(self.input_features_versions).needed_entities()

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        from ollama import AsyncClient
        import polars as pl

        client = AsyncClient(host=self.endpoint)

        expected_cols = [
            feat.name
            for feat in store.feature_references_for(self.input_features_versions)
        ]

        entities = await values.to_polars()

        prompts = []

        if self.precomputed_prompt_key_overwrite in entities.columns:
            prompts = entities[self.precomputed_prompt_key_overwrite].to_list()
        else:
            missing_cols = set(expected_cols) - set(entities.columns)
            if missing_cols:
                entities = (
                    await store.using_version(self.input_features_versions)
                    .features_for(values)
                    .with_subfeatures()
                    .to_polars()
                )

            for index, value in enumerate(entities.iter_rows(named=True)):
                logger.info(f"Processing row {index + 1}/{len(prompts)}")

                prompt = self.prompt_template.format(**value)
                prompts.append(prompt)

            entities = entities.with_columns(
                pl.Series(name=self.precomputed_prompt_key_overwrite, values=prompts)
            )

        ret_vals = []

        for prompt in prompts:
            response = await client.embeddings(self.model_name, prompt)

            if isinstance(response, dict):
                embedding = response["embedding"]  # type: ignore
            else:
                embedding = response

            ret_vals.append(embedding)

        pred_view = store.model.predictions_view
        if pred_view.model_version_column:
            model_version = f"{self.prompt_template_hash()} -> {self.model_name}"
            model_version_name = pred_view.model_version_column.name
            entities = entities.with_columns(
                pl.lit(model_version).alias(model_version_name)
            )

        if pred_view.event_timestamp:
            new_et = pred_view.event_timestamp.name
            existing_et = values.request_result.event_timestamp
            need_to_add_et = new_et not in entities.columns

            if existing_et and need_to_add_et and existing_et in entities.columns:
                logger.info(
                    f"Using existing event timestamp `{existing_et}` as new timestamp."
                )
                entities = entities.with_columns(pl.col(existing_et).alias(new_et))
            elif need_to_add_et:
                logger.info("No event timestamp using now as the timestamp.")
                entities = entities.with_columns(
                    pl.lit(datetime.now(tz=timezone.utc)).alias(
                        pred_view.event_timestamp.name
                    )
                )

        return entities.hstack([pl.Series(name=self.embedding_name, values=ret_vals)])


def ollama_generate(
    endpoint: str,
    model: str,
    prompt_template: str,
    input_features_versions: str,
) -> "OllamaGeneratePredictor":
    return OllamaGeneratePredictor(
        endpoint=endpoint,
        model_name=model,
        prompt_template=prompt_template,
        input_features_versions=input_features_versions,
    )


def ollama_embedding(
    endpoint: str,
    model: str,
    input_features_versions: str,
    prompt_template: str,
    embedding_name: str | None = None,
    precomputed_prompt_key: str = "full_prompt",
) -> "OllamaEmbeddingPredictor":
    return OllamaEmbeddingPredictor(
        endpoint=endpoint,
        model_name=model,
        prompt_template=prompt_template,
        input_features_versions=input_features_versions,
        embedding_name=embedding_name or "embedding",
        precomputed_prompt_key_overwrite=precomputed_prompt_key,
    )


class OllamaGeneration:
    model: String

    response: String
    created_at: EventTimestamp

    context: List

    load_duration: Int64
    total_duration: Int64

    prompt_eval_count: Int32
    prompt_eval_duration: Int64

    eval_count: Int32
    eval_duration: Int64


def ollama_generate_contract(
    prompt: FeatureReferencable,
    contract_name: str,
    endpoint: str,
    model: str,
    entities: list[FeatureFactory] | FeatureFactory,
    output_source: CodableBatchDataSource | None = None,
    contacts: list[str] | None = None,
    tags: list[str] | None = None,
) -> ModelContractWrapper[OllamaGeneration]:
    from aligned import model_contract, ExposedModel

    @model_contract(
        name=contract_name,
        description=f"Contract for generating text using the {model} through Ollama.",
        input_features=[prompt],
        exposed_model=ExposedModel.ollama_generate(
            endpoint=endpoint,
            model=model,
            prompt_template=f"{{{prompt.feature_reference().name}}}",
            input_features_versions="default",
        ),
        output_source=output_source,
        tags=tags,
        contacts=contacts,
    )
    class OllamaOutput:
        model = String().as_model_version()

        input_prompt = String()

        response = String()
        created_at = EventTimestamp()

        context = List(Int32())

        load_duration = Int64()
        total_duration = Int64()

        prompt_eval_count = Int32()
        prompt_eval_duration = Int64()

        eval_count = Int32()
        eval_duration = Int64()

    if not isinstance(entities, list):
        entities = [entities]

    for entity in entities:
        feature = entity.copy_type()
        assert isinstance(feature, CouldBeEntityFeature)
        feature = feature.as_entity()
        feature._name = entity.name

        setattr(OllamaOutput.contract, entity.name, feature)

    return OllamaOutput  # type: ignore
