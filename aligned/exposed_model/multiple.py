from __future__ import annotations

import polars as pl
from typing import TYPE_CHECKING
from datetime import datetime, timezone
from dataclasses import dataclass
from aligned.retrieval_job import RetrievalJob
from aligned.exposed_model.interface import ExposedModel, VersionedModel
import logging

from aligned.schemas.feature import (
    Feature,
    FeatureReference,
)

if TYPE_CHECKING:
    from aligned.feature_store import ModelFeatureStore

logger = logging.getLogger(__name__)


@dataclass
class MultipleModels(ExposedModel):
    """
    A model that runs multiple models in parallel and returns the
    result of one of them based on a weight and a random value.
    """

    models: list[ExposedModel]

    model_type: str = "mult_models"

    @property
    def exposed_at_url(self) -> str | None:
        return self.models[0].exposed_at_url

    @property
    def as_markdown(self) -> str:
        model_definitions = [
            f"**Model {i}**: {model.as_markdown}."
            for i, model in enumerate(self.models)
        ]
        return "\n\n".join(model_definitions)

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        features = []
        for model in self.models:
            features += await model.needed_features(store)
        return features

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        entities: set[Feature] = set()
        for model in self.models:
            entities = entities.union(await model.needed_entities(store))
        return entities

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        pred_view = store.model.predictions_view
        pred_feature = pred_view.model_version_column
        timestamp_feature = pred_view.event_timestamp

        all_columns = store.prediction_request().all_returned_columns

        async def format_preds(
            preds: pl.DataFrame, model: ExposedModel
        ) -> pl.DataFrame:
            if pred_feature:
                version = "unknown"
                if isinstance(model, VersionedModel):
                    version = await model.model_version()

                preds = preds.with_columns(pl.lit(version).alias(pred_feature.name))

            if timestamp_feature:
                preds = preds.with_columns(
                    pl.lit(datetime.now(tz=timezone.utc)).alias(timestamp_feature.name)
                )

            return preds.select(all_columns)

        preds = await format_preds(
            await self.models[0].run_polars(values, store), self.models[0]
        )

        for model in self.models[1:]:
            new_preds = await model.run_polars(values, store)

            preds = preds.vstack(await format_preds(new_preds, model))

        return preds
