from dataclasses import dataclass
import polars as pl

from aligned.exposed_model.interface import ExposedModel
from aligned.feature_store import ModelFeatureStore
from aligned.schemas.feature import Feature, FeatureReference
from aligned.retrival_job import RetrivalJob


@dataclass
class PartitionedModel(ExposedModel):

    partition_key: Feature
    partitions: dict[str, ExposedModel]

    default_partition: str | None

    model_type = 'partitioned'

    @property
    def as_markdown(self) -> str:
        return f"""### Partitioned model
Partitione key: {self.partition_key.name}

Partitions: {self.partitions}"""

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:

        all_features: set[FeatureReference] = set()

        for model in self.partitions.values():
            all_features.update(await model.needed_features(store))

        return list(all_features)

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        all_features = await self.needed_features(store)
        entities = store.store.needed_entities_for(all_features)
        entities.add(self.partition_key)
        return entities

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        df = await values.to_lazy_polars()

        partitions = df.select(partition=pl.col(self.partition_key.name).unique()).collect()

        outputs: pl.DataFrame | None = None

        for partition_value in partitions['partition'].to_list():

            model = self.partitions.get(partition_value)

            if model is None and self.default_partition:
                model = self.partitions.get(self.default_partition)

            if model is None:
                raise ValueError(
                    f"Unable to find model for partition {partition_value} for model {store.model.name}.\n\n"
                    'Either add an additional model, or a default partition to use.'
                )

            subset = df.filter(pl.col(self.partition_key.name) == partition_value)
            if outputs is None:
                outputs = await model.run_polars(
                    RetrivalJob.from_polars_df(subset, values.retrival_requests), store
                )
            else:
                preds = await model.run_polars(
                    RetrivalJob.from_polars_df(subset, values.retrival_requests), store
                )
                outputs = pl.concat([outputs, preds.select(outputs.columns)], how='vertical_relaxed')

        assert outputs is not None
        return outputs
