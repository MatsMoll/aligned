import logging
from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import polars as pl

from aligned.compiler.feature_factory import EventTimestamp, FeatureFactory, Target, TargetProbability
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.entity_data_source import EntityDataSource
from aligned.feature_view.feature_view import FeatureView
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReferance, FeatureType
from aligned.schemas.literal_value import LiteralValue
from aligned.schemas.model import EventTrigger, InferenceView
from aligned.schemas.model import Model as ModelSchema
from aligned.schemas.model import Target as TargetSchema

logger = logging.getLogger(__name__)


class SqlEntityDataSource(EntityDataSource):

    url: str
    timestamp_column: str

    def __init__(self, sql: Callable[[str], str], url: str, timestamp_column: str) -> None:
        self.sql = sql
        self.url = url
        self.timestamp_column = timestamp_column

    async def all_in_range(self, start_date: datetime, end_date: datetime) -> pl.DataFrame:
        import os

        start = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end = end_date.strftime('%Y-%m-%d %H:%M:%S')

        query = self.sql(f'{self.timestamp_column} BETWEEN \'{start}\' AND \'{end}\'')
        return pl.read_sql(query, os.environ[self.url])

    async def last(self, days: int, hours: int, seconds: int) -> pl.DataFrame:
        now = datetime.utcnow()
        return await self.all_in_range(now - timedelta(days=days, hours=hours, seconds=seconds), now)


@dataclass
class ModelMetedata:
    name: str
    features: list[FeatureFactory]
    # Will log the feature inputs to a model. Therefore, enabling log and wait etc.
    # feature_logger: WritableBatchSource | None = field(default=None)
    inference_source: BatchDataSource | None = field(default=None)
    inference_stream: StreamDataSource | None = field(default=None)


class Model(ABC):
    @staticmethod
    def metadata_with(
        name: str,
        description: str,
        features: list[FeatureFactory],
        inference_source: BatchDataSource | None = None,
        inference_stream: StreamDataSource | None = None,
    ) -> ModelMetedata:
        return ModelMetedata(name, features, inference_source, inference_stream)

    @abstractproperty
    def metadata(self) -> ModelMetedata:
        pass

    @classmethod
    def compile(cls) -> ModelSchema:
        var_names = [name for name in cls().__dir__() if not name.startswith('_')]
        metadata = cls().metadata

        inference_view: InferenceView = InferenceView(
            set(), set(), set(), set(), source=metadata.inference_source
        )
        probability_features: dict[str, set[TargetProbability]] = {}

        for var_name in var_names:
            feature = getattr(cls, var_name)

            if isinstance(feature, FeatureFactory):
                feature._name = var_name
                feature._location = FeatureLocation(metadata.name, 'model')

            if isinstance(feature, FeatureView):
                compiled = feature.compile()
                inference_view.entities.update(compiled.entities)
            elif isinstance(feature, Target):
                feature._name = var_name
                target_feature = feature.feature.copy_type()
                target_feature._name = var_name
                trigger: EventTrigger | None = None

                if feature.event_trigger:
                    event = feature.event_trigger
                    if not event.condition._name:
                        event.condition._name = '0'

                    trigger = EventTrigger(
                        event.condition.compile(), event=event.event, payload={feature.feature.feature()}
                    )
                inference_view.target.add(
                    TargetSchema(
                        estimating=feature.reference(),
                        feature=target_feature.feature(),
                        event_trigger=trigger,
                    )
                )
            elif isinstance(feature, EventTimestamp):
                inference_view.event_timestamp = feature.event_timestamp()

            elif isinstance(feature, TargetProbability):
                feature_name = feature.target._name
                feature._name = var_name
                inference_view.features.add(
                    Feature(
                        var_name,
                        FeatureType('').float,
                        f"The probability of target named {feature_name} being '{feature.of_value}'.",
                    )
                )
                probability_features[feature_name] = probability_features.get(feature_name, set()).union(
                    {feature}
                )
            elif isinstance(feature, FeatureFactory):
                inference_view.features.add(feature.feature())

        # Needs to run after the feature views have compiled
        features: set[FeatureReferance] = {feature.feature_referance() for feature in metadata.features}

        for target, probabilities in probability_features.items():
            from aligned.schemas.transformation import MapArgMax

            transformation = MapArgMax(
                {probs._name: LiteralValue.from_value(probs.of_value) for probs in probabilities}
            )

            arg_max_feature = DerivedFeature(
                name=target,
                dtype=transformation.dtype,
                transformation=transformation,
                depending_on={
                    FeatureReferance(
                        feat, FeatureLocation(metadata.name, 'model'), dtype=FeatureType('').float
                    )
                    for feat in transformation.column_mappings.keys()
                },
                depth=1,
            )
            inference_view.derived_features.add(arg_max_feature)
        if not probability_features:
            inference_view.features.update({target.feature for target in inference_view.target})

        return ModelSchema(name=metadata.name, features=features, inference_view=inference_view)
