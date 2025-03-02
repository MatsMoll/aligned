from dataclasses import dataclass
import polars as pl

from aligned.config_value import ConfigValue
from aligned.exposed_model.interface import ExposedModel
from aligned.feature_store import ModelFeatureStore
from aligned.schemas.feature import Feature, FeatureReference
from aligned.retrieval_job import RetrievalJob


@dataclass
class PartitionedModel(ExposedModel):
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

    partition_key: Feature
    partitions: dict[str, ExposedModel]

    default_partition: str | None

    model_type = "partitioned"

    @property
    def as_markdown(self) -> str:
        return f"""### Partitioned model
Partitione key: {self.partition_key.name}

Partitions: {self.partitions}"""

    def needed_configs(self) -> list[ConfigValue]:
        configs = []
        for partition in self.partitions.values():
            configs.extend(partition.needed_configs())
        return configs

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

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        df = await values.to_lazy_polars()

        partitions = df.select(
            partition=pl.col(self.partition_key.name).unique()
        ).collect()

        outputs: pl.DataFrame | None = None

        for partition_value in partitions["partition"].to_list():
            model = self.partitions.get(partition_value)

            if model is None and self.default_partition:
                model = self.partitions.get(self.default_partition)

            if model is None:
                raise ValueError(
                    f"Unable to find model for partition {partition_value} for model {store.model.name}.\n\n"
                    "Either add an additional model, or a default partition to use."
                )

            subset = df.filter(pl.col(self.partition_key.name) == partition_value)
            if outputs is None:
                outputs = await model.run_polars(
                    RetrievalJob.from_polars_df(subset, values.retrieval_requests),
                    store,
                )
            else:
                preds = await model.run_polars(
                    RetrievalJob.from_polars_df(subset, values.retrieval_requests),
                    store,
                )
                outputs = pl.concat(
                    [outputs, preds.select(outputs.columns)], how="vertical_relaxed"
                )

        assert outputs is not None
        return outputs
