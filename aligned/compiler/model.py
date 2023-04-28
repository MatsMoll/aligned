import logging
from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import polars as pl

from aligned.compiler.feature_factory import (
    ClassificationTarget,
    EventTimestamp,
    FeatureFactory,
    FeatureReferencable,
    RegressionTarget,
    TargetProbability,
)
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.entity_data_source import EntityDataSource
from aligned.feature_view.feature_view import FeatureView
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReferance, FeatureType
from aligned.schemas.folder import Folder
from aligned.schemas.literal_value import LiteralValue
from aligned.schemas.model import Model as ModelSchema
from aligned.schemas.model import PredictionsView
from aligned.schemas.target import ClassificationTarget as ClassificationTargetSchema
from aligned.schemas.target import RegressionTarget as RegressionTargetSchema

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
    features: list[FeatureReferencable]
    # Will log the feature inputs to a model. Therefore, enabling log and wait etc.
    # feature_logger: WritableBatchSource | None = field(default=None)
    contacts: list[str] | None = field(default=None)
    tags: dict[str, str] | None = field(default=None)
    description: str | None = field(default=None)
    predictions_source: BatchDataSource | None = field(default=None)
    predictions_stream: StreamDataSource | None = field(default=None)
    dataset_folder: Folder | None = field(default=None)


class Model(ABC):
    @staticmethod
    def metadata_with(
        name: str,
        description: str,
        features: list[FeatureReferencable],
        contacts: list[str] | None = None,
        tags: dict[str, str] | None = None,
        predictions_source: BatchDataSource | None = None,
        predictions_stream: StreamDataSource | None = None,
        dataset_folder: Folder | None = None,
    ) -> ModelMetedata:
        return ModelMetedata(
            name,
            features,
            contacts,
            tags,
            description,
            predictions_source,
            predictions_stream,
            dataset_folder,
        )

    @abstractproperty
    def metadata(self) -> ModelMetedata:
        pass

    @classmethod
    def compile(cls) -> ModelSchema:
        var_names = [name for name in cls().__dir__() if not name.startswith('_')]
        metadata = cls().metadata

        inference_view: PredictionsView = PredictionsView(
            entities=set(),
            features=set(),
            derived_features=set(),
            source=metadata.predictions_source,
            stream_source=metadata.predictions_stream,
            classification_targets=set(),
            regression_targets=set(),
        )
        probability_features: dict[str, set[TargetProbability]] = {}

        classification_targets: dict[str, ClassificationTargetSchema] = {}
        regression_targets: dict[str, RegressionTargetSchema] = {}

        for var_name in var_names:
            feature = getattr(cls, var_name)

            if isinstance(feature, FeatureFactory):
                feature._location = FeatureLocation.model(metadata.name)

            if isinstance(feature, FeatureView):
                compiled = feature.compile()
                inference_view.entities.update(compiled.entities)
            elif isinstance(feature, Model):
                compiled = feature.compile()
                inference_view.entities.update(compiled.predictions_view.entities)
            elif isinstance(feature, ClassificationTarget):
                assert feature._name
                feature._location = FeatureLocation.model(metadata.name)
                target_feature = feature.compile()

                classification_targets[var_name] = target_feature
                inference_view.classification_targets.add(target_feature)
            elif isinstance(feature, RegressionTarget):
                assert feature._name
                feature._location = FeatureLocation.model(metadata.name)
                target_feature = feature.compile()

                regression_targets[var_name] = target_feature
                inference_view.regression_targets.add(target_feature)
            elif isinstance(feature, EventTimestamp):
                inference_view.event_timestamp = feature.event_timestamp()

            elif isinstance(feature, TargetProbability):
                feature_name = feature.target._name
                assert feature._name
                assert (
                    feature.target._name in classification_targets
                ), 'Target must be a classification target.'

                target = classification_targets[feature.target._name]
                target.class_probabilities.add(feature.compile())

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
                    FeatureReferance(feat, FeatureLocation.model(metadata.name), dtype=FeatureType('').float)
                    for feat in transformation.column_mappings.keys()
                },
                depth=1,
            )
            inference_view.derived_features.add(arg_max_feature)
        if not probability_features:
            inference_view.features.update(
                {target.feature for target in inference_view.classification_targets}
            )

        return ModelSchema(
            name=metadata.name,
            features=features,
            predictions_view=inference_view,
            contacts=metadata.contacts,
            tags=metadata.tags,
            description=metadata.description,
            dataset_folder=metadata.dataset_folder,
        )
