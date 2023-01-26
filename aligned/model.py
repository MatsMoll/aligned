import logging
from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import polars as pl

from aligned.compiler.feature_factory import EventTimestamp, Target
from aligned.data_source.batch_data_source import BatchDataSource
from aligned.entity_data_source import EntityDataSource
from aligned.request.retrival_request import FeatureRequest
from aligned.schemas.feature import FeatureReferance
from aligned.schemas.model import EventTrigger
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
    features: list[FeatureRequest]
    # Will log the feature inputs to a model. Therefore, enabling log and wait etc.
    # feature_logger: WritableBatchSource | None = field(default=None)
    inference_source: BatchDataSource | None = field(default=None)


class Model(ABC):
    @staticmethod
    def metadata_with(
        name: str, features: list[FeatureRequest], inference_source: BatchDataSource | None = None
    ) -> ModelMetedata:
        return ModelMetedata(name, features, inference_source)

    @abstractproperty
    def metadata(self) -> ModelMetedata:
        pass

    @classmethod
    def compile(cls) -> ModelSchema:
        var_names = [name for name in cls().__dir__() if not name.startswith('_')]
        metadata = cls().metadata

        targets: set[TargetSchema] = set()
        features: set[FeatureReferance] = set()
        timestamp: EventTimestamp | None = None

        for request in metadata.features:
            features.update(
                {
                    FeatureReferance(feature.name, request.name, feature.dtype)
                    for feature in request.request_result.features
                    if feature.name in request.features_to_include
                }
            )

        for var_name in var_names:
            feature = getattr(cls, var_name)

            if isinstance(feature, Target):
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
                targets.add(
                    TargetSchema(
                        estimating=feature.reference(),
                        feature=target_feature.feature(),
                        event_trigger=trigger,
                    )
                )
            elif isinstance(feature, EventTimestamp):
                feature._name = var_name
                timestamp = feature.event_timestamp()

        return ModelSchema(
            name=metadata.name,
            features=features,
            targets=targets,
            inference_source=metadata.inference_source,
            event_timestamp=timestamp,
        )
