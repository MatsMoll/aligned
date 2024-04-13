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


logger = logging.getLogger(__name__)


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
    prompt: FeatureFactory,
    contract_name: str,
    endpoint: str,
    model: str,
    entities: list[FeatureFactory] | FeatureFactory,
    prediction_source: BatchDataSource | None = None,
) -> ModelContractWrapper[OllamaGeneration]:
    from aligned import model_contract, ExposedModel

    @model_contract(
        name=contract_name,
        input_features=[prompt],
        exposed_model=ExposedModel.ollama_generate(
            endpoint=endpoint,
            model=model,
            prompt_template=f"{{{prompt.name}}}",
            input_features_versions='default',
        ),
        output_source=prediction_source,
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
    input: FeatureFactory | list[FeatureFactory],
    contract_name: str,
    endpoint: str,
    model: str,
    entities: list[FeatureFactory] | FeatureFactory,
    output_source: BatchDataSource | None = None,
    prompt_template: str | None = None,
):
    from aligned import model_contract, FeatureInputVersions

    if isinstance(input, FeatureFactory) and prompt_template is None:
        prompt_template = f"{{{input.name}}}"

    if prompt_template is None:
        raise ValueError('prompt_template must be provided if input is a list')

    if not isinstance(input, list):
        input = [input]

    @model_contract(
        name=contract_name,
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
    )
    class OllamaEmbedding:

        embedding = Embedding()

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
        input_features=input,
        exposed_model=ExposedModel.ollama_generate(
            endpoint=endpoint, model=model, prompt_template=prompt_template, input_features_versions='default'
        ),
        output_source=output_source,
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
