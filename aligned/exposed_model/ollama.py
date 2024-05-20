from dataclasses import dataclass
from aligned.compiler.model import ModelContractWrapper
from aligned.compiler.feature_factory import (
    Embedding,
    Entity,
    FeatureFactory,
    FeatureReferencable,
    Int32,
    String,
    List,
    Int64,
    EventTimestamp,
)
import logging

from aligned.data_source.batch_data_source import BatchDataSource
from aligned.exposed_model.interface import ExposedModel
from aligned.schemas.feature import Feature, FeatureReference
from aligned.retrival_job import RetrivalJob
import polars as pl
from aligned.feature_store import ModelFeatureStore


logger = logging.getLogger(__name__)


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

    async def potential_drift_from_model(self, old_model: ExposedModel) -> str | None:
        """
        Checks if a change in model can lead to a potential distribution shift.

        Returns:
            str: A message explaining the potential drift.
        """
        if not isinstance(old_model, OllamaEmbeddingPredictor):
            return None

        changes = ''
        if old_model.model_name != self.model_name:
            changes += f"Model name changed from `{old_model.model_name}` to `{self.model_name}`.\n"

        if old_model.prompt_template != self.prompt_template:
            changes += (
                f"Prompt template changed from `{old_model.prompt_template}` to `{self.prompt_template}`.\n"
            )

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

        for index, value in enumerate(prompts.iter_rows(named=True)):
            logger.info(f"Processing row {index + 1}/{len(prompts)}")

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
    output_source: BatchDataSource | None = None,
    contacts: list[str] | None = None,
    tags: list[str] | None = None,
) -> ModelContractWrapper[OllamaGeneration]:
    from aligned import model_contract, ExposedModel

    @model_contract(
        name=contract_name,
        description=f'Contract for generating text using the {model} through Ollama.',
        input_features=[prompt],
        exposed_model=ExposedModel.ollama_generate(
            endpoint=endpoint,
            model=model,
            prompt_template=f"{{{prompt.feature_reference().name}}}",
            input_features_versions='default',
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
        if isinstance(entity, Entity):
            feature = entity._dtype.copy_type()
        else:
            feature = entity.copy_type()

        new_entity = Entity(feature)

        feature._name = entity.name
        new_entity._name = entity.name

        setattr(OllamaOutput.contract, entity.name, new_entity)

    return OllamaOutput  # type: ignore


def ollama_embedding_contract(
    input: FeatureReferencable | list[FeatureReferencable],
    contract_name: str,
    endpoint: str,
    model: str,
    entities: list[FeatureFactory] | FeatureFactory,
    output_source: BatchDataSource | None = None,
    prompt_template: str | None = None,
    embedding_size: int | None = None,
    contacts: list[str] | None = None,
    tags: list[str] | None = None,
):
    from aligned import model_contract, FeatureInputVersions

    if isinstance(input, FeatureReferencable) and prompt_template is None:
        ref = input.feature_reference()
        prompt_template = f"{{{ref.name}}}"

    if prompt_template is None:
        raise ValueError('prompt_template must be provided if input is a list')

    if not isinstance(input, list):
        input = [input]

    emb_size = embedding_size

    ollama_model_sizes = {
        'nomic-embed-text': 768,
        'mxbai-embed-large': 512,
        'all-minilm': 768,
    }

    if not emb_size:
        emb_size = ollama_model_sizes.get(model)

    if not emb_size:
        raise ValueError('embedding_size must be provided if model is not in the default sizes')

    @model_contract(
        name=contract_name,
        description=f'Contract for generating embeddings using the {model} through Ollama',
        input_features=FeatureInputVersions(
            default_version='default', versions={'default': input}  # type: ignore
        ),
        exposed_model=ExposedModel.ollama_embedding(
            endpoint=endpoint,
            model=model,
            input_features_versions='default',
            prompt_template=prompt_template,
            embedding_name='embedding',
        ),
        output_source=output_source,
        contacts=contacts,
        tags=tags,
    )
    class OllamaEmbedding:

        embedding = Embedding(embedding_size=emb_size)

    if not isinstance(entities, list):
        entities = [entities]

    for entity in entities:
        if isinstance(entity, Entity):
            feature = entity._dtype.copy_type()
        else:
            feature = entity.copy_type()

        new_entity = Entity(feature)

        feature._name = entity.name
        new_entity._name = entity.name

        setattr(OllamaEmbedding.contract, entity.name, new_entity)

    return OllamaEmbedding  # type: ignore


def ollama_classification_contract(
    input: list[FeatureReferencable] | FeatureReferencable,
    contract_name: str,
    endpoint: str,
    model: str,
    entities: list[FeatureFactory] | FeatureFactory,
    ground_truth: FeatureFactory,
    output_source: BatchDataSource | None = None,
    prompt_template: str | None = None,
    contacts: list[str] | None = None,
    tags: list[str] | None = None,
):
    from aligned import model_contract, ExposedModel
    from aligned.schemas.constraints import InDomain

    if not isinstance(input, list):
        input = [input]

    allowed_outputs = []

    if ground_truth.constraints:
        for constraint in ground_truth.constraints:
            if isinstance(constraint, InDomain):
                allowed_outputs = constraint.values

    if not prompt_template:
        prompt_template = ''

        if allowed_outputs:
            prompt_template = (
                "Your task is to classify the input into one of the following classes: '"
                + "', '".join(allowed_outputs)
                + "'.\n\n"
            )

        prompt_template += 'You have the following information at your disposal:\n'

        for feature in input:
            ref = feature.feature_reference()
            prompt_template += f"{ref.name}: {{{ref.name}}}\n"

        prompt_template += (
            '\n\nDo not explain why you think the input belongs to a certain class, '
            'just provide the class you think the input belongs to. '
            'If you are unsure about which class it belong to, return `Unknown`.'
        )

    @model_contract(
        name=contract_name,
        description=f'Contract for classifying text using the {model} through Ollama',
        input_features=input,  # type: ignore
        exposed_model=ExposedModel.ollama_generate(
            endpoint=endpoint, model=model, prompt_template=prompt_template, input_features_versions='default'
        ),
        output_source=output_source,
        contacts=contacts,
        tags=tags,
    )
    class OllamaOutput:
        model_version = (
            String()
            .as_model_version()
            .description('This is a combination of the used LLM model, and the prompt template.')
        )
        created_at = EventTimestamp()
        response = ground_truth.as_classification_label()

    if not isinstance(entities, list):
        entities = [entities]

    for entity in entities:
        if isinstance(entity, Entity):
            feature = entity._dtype.copy_type()
        else:
            feature = entity.copy_type()

        new_entity = Entity(feature)

        feature._name = entity.name
        new_entity._name = entity.name

        setattr(OllamaOutput.contract, entity.name, new_entity)

    return OllamaOutput  # type: ignore
