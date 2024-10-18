from dataclasses import dataclass
import polars as pl

from aligned.exposed_model.interface import ExposedModel
from aligned.feature_store import ModelFeatureStore
from aligned.retrival_job import RetrivalJob
from aligned.schemas.feature import Feature, FeatureReference
from aligned.schemas.model import Model


@dataclass
class OpenAiEmbeddingPredictor(ExposedModel):

    model: str

    feature_refs: list[FeatureReference] = None  # type: ignore
    output_name: str = ''
    prompt_template: str = ''
    model_type = 'openai_emb'

    @property
    def exposed_at_url(self) -> str | None:
        return 'https://api.openai.com/'

    def prompt_template_hash(self) -> str:
        from hashlib import sha256

        return sha256(self.prompt_template.encode(), usedforsecurity=False).hexdigest()

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return self.feature_refs

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.store.requests_for_features(self.feature_refs).request_result.entities

    def with_contract(self, model: Model) -> ExposedModel:
        embeddings = model.predictions_view.embeddings()
        assert len(embeddings) == 1, f"Need at least one embedding. Got {len(embeddings)}"

        if self.output_name == '':
            self.output_name = embeddings[0].name

        if not self.feature_refs:
            self.feature_refs = list(model.feature_references())

        if self.prompt_template == '':
            for feat in self.feature_refs:
                self.prompt_template += f"{feat.name}: {{{feat.name}}}\n\n"

        return self

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        from openai import AsyncClient

        client = AsyncClient()

        expected_cols = {feat.name for feat in self.feature_refs}
        missing_cols = expected_cols - set(values.loaded_columns)

        if missing_cols:
            df = await store.store.features_for(values, features=self.feature_refs).to_polars()
        else:
            df = await values.to_polars()

        if len(expected_cols) == 1:
            prompts = df[self.feature_refs[0].name].to_list()
        else:
            prompts: list[str] = []
            for row in df.to_dicts():
                prompts.append(self.prompt_template.format(**row))

        embeddings = await client.embeddings.create(input=prompts, model=self.model)
        return df.hstack(
            [
                pl.Series(
                    name=self.output_name,
                    values=[emb.embedding for emb in embeddings.data],
                    dtype=pl.List(pl.Float32),
                )
            ]
        )
