import logging
import polars as pl
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from aligned.compiler.feature_factory import (
    CouldBeEntityFeature,
    CouldBeModelVersion,
    Embedding,
    EventTimestamp,
    FeatureFactory,
    FeatureReferencable,
    ModelVersion,
    String,
)
from aligned.data_source.batch_data_source import CodableBatchDataSource
from aligned.feature_store import ModelFeatureStore

from aligned.exposed_model.interface import ExposedModel, PromptModel
from aligned.retrival_job import RetrivalJob
from aligned.schemas.feature import Feature, FeatureReference


logger = logging.getLogger(__name__)


@dataclass
class SentenceTransformerPredictor(ExposedModel, PromptModel):

    model_name: str
    embedding_name: str

    prompt_template: str
    features_to_load: list[FeatureReference]

    precomputed_prompt_key_overwrite: str

    model_type: str = 'sent_tran'

    @property
    def precomputed_prompt_key(self) -> str | None:
        return self.precomputed_prompt_key_overwrite

    @property
    def exposed_at_url(self) -> str | None:
        return None

    def prompt_template_hash(self) -> str:
        from hashlib import sha256

        return sha256(self.prompt_template.encode(), usedforsecurity=False).hexdigest()

    async def potential_drift_from_model(self, old_model: ExposedModel) -> str | None:
        """
        Checks if a change in model can lead to a potential distribution shift.

        Returns:
            str: A message explaining the potential drift.
        """
        if not isinstance(old_model, SentenceTransformerPredictor):
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
        return f"""Creating an embedding through a sentence transformer.

This will use the model: `{self.model_name}` to generate the embeddings."""

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return self.features_to_load

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.store.requests_for_features(self.features_to_load).request_result.entities

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        import polars as pl
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name_or_path=self.model_name)
        entities = await values.to_polars()

        prompts: list[str] = []

        if self.precomputed_prompt_key_overwrite in entities.columns:
            prompts = entities[self.precomputed_prompt_key_overwrite].to_list()
        else:
            missing_cols = {feat.name for feat in self.features_to_load} - set(entities.columns)
            if missing_cols:
                entities = await store.store.features_for(values, self.features_to_load).to_polars()

            for index, value in enumerate(entities.iter_rows(named=True)):
                logger.info(f"Processing row {index + 1}/{len(prompts)}")

                prompt = self.prompt_template.format(**value)
                prompts.append(prompt)

            entities = entities.with_columns(
                pl.Series(name=self.precomputed_prompt_key_overwrite, values=prompts)
            )

        responses = model.encode(prompts)

        pred_view = store.model.predictions_view
        if pred_view.model_version_column:
            model_version = f"{self.prompt_template_hash()} -> {self.model_name}"
            model_version_name = pred_view.model_version_column.name
            entities = entities.with_columns(pl.lit(model_version).alias(model_version_name))

        if pred_view.event_timestamp:
            new_et = pred_view.event_timestamp.name
            existing_et = values.request_result.event_timestamp
            need_to_add_et = new_et not in entities.columns

            if existing_et and need_to_add_et and existing_et in entities.columns:
                logger.info(f"Using existing event timestamp `{existing_et}` as new timestamp.")
                entities = entities.with_columns(pl.col(existing_et).alias(new_et))
            elif need_to_add_et:
                logger.info('No event timestamp using now as the timestamp.')
                entities = entities.with_columns(
                    pl.lit(datetime.now(tz=timezone.utc)).alias(pred_view.event_timestamp.name)
                )

        return entities.hstack([pl.Series(name=self.embedding_name, values=responses)])


def sentence_transform_feature(
    feature: FeatureReferencable,
    model: str,
    output_name: str,
    precomputed_prompt_key: str = 'full_prompt',
) -> 'SentenceTransformerPredictor':

    return SentenceTransformerPredictor(
        model_name=model,
        embedding_name=output_name,
        prompt_template=f"{{{feature.feature_reference().name}}}",
        features_to_load=[feature.feature_reference()],
        precomputed_prompt_key_overwrite=precomputed_prompt_key,
    )


def sentence_transform_prompt(
    model: str,
    features: list[FeatureReferencable],
    prompt_template: str,
    output_name: str,
    precomputed_prompt_key: str = 'full_prompt',
) -> 'SentenceTransformerPredictor':

    return SentenceTransformerPredictor(
        model_name=model,
        embedding_name=output_name,
        prompt_template=prompt_template,
        features_to_load=[feature.feature_reference() for feature in features],
        precomputed_prompt_key_overwrite=precomputed_prompt_key,
    )


def sentence_transformer_contract(
    input: FeatureReferencable | list[FeatureReferencable],
    contract_name: str,
    model: str,
    entities: list[FeatureFactory] | FeatureFactory,
    embedding_size: int,
    prompt_template: str | None = None,
    output_source: CodableBatchDataSource | None = None,
    contacts: list[str] | None = None,
    tags: list[str] | None = None,
    precomputed_prompt_key: str = 'full_prompt',
    model_version_field: FeatureFactory | None = None,
    additional_metadata: list[FeatureFactory] | None = None,
    acceptable_freshness: timedelta | None = None,
    unacceptable_freshness: timedelta | None = None,
):
    from aligned import model_contract, FeatureInputVersions

    if isinstance(input, FeatureReferencable) and prompt_template is None:
        ref = input.feature_reference()
        prompt_template = f"{{{ref.name}}}"

    if prompt_template is None:
        raise ValueError('prompt_template must be provided if input is a list')

    if not isinstance(input, list):
        input = [input]

    @model_contract(
        name=contract_name,
        description=f'Contract for generating embeddings using the {model} through Ollama',
        input_features=FeatureInputVersions(
            default_version='default', versions={'default': input}  # type: ignore
        ),
        exposed_model=sentence_transform_prompt(
            model=model,
            features=input,
            prompt_template=prompt_template,
            output_name='embedding',
            precomputed_prompt_key=precomputed_prompt_key,
        ),
        output_source=output_source,
        contacts=contacts,
        tags=tags,
        acceptable_freshness=acceptable_freshness,
        unacceptable_freshness=unacceptable_freshness,
    )
    class ScentenceTransformerContract:
        updated_at = EventTimestamp()
        embedding = Embedding(embedding_size=embedding_size)
        full_prompt = String().with_name(precomputed_prompt_key)

    if not isinstance(entities, list):
        entities = [entities]

    for entity in entities:
        feature = entity.copy_type()
        assert isinstance(feature, CouldBeEntityFeature)
        new_entity = feature.as_entity()

        feature._name = entity.name
        new_entity._name = entity.name

        setattr(ScentenceTransformerContract.contract, entity.name, new_entity)

    def add_feature(feature: FeatureFactory) -> None:

        assert feature._name, (
            'Trying to add a feature without any name. '
            'Consider using the `.with_name(...)` to manually set it.'
        )
        if feature._location is None:
            setattr(ScentenceTransformerContract.contract, feature.name, feature)
            return

        feature_copy = feature.copy_type()
        feature_copy._name = feature._name
        feature_copy.constraints = feature.constraints.copy() if feature.constraints else None
        setattr(ScentenceTransformerContract.contract, feature_copy.name, feature_copy)

    if model_version_field is not None:
        if isinstance(model_version_field, ModelVersion):
            add_feature(model_version_field)
        elif isinstance(model_version_field, CouldBeModelVersion):
            add_feature(model_version_field.as_model_version().with_name(model_version_field.name))
        else:
            raise ValueError(f"Feature {model_version_field} can not be a model version.")
    else:
        add_feature(String().as_model_version().with_name('model_version'))

    for feature in additional_metadata or []:
        add_feature(feature)

    return ScentenceTransformerContract  # type: ignore
