import logging
from dataclasses import dataclass

import pandas as pd
import polars as pl

from aligned.data_source.stream_data_source import StreamDataSource
from aligned.retrival_job import RequestResult
from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import Feature

logger = logging.getLogger(__name__)


@dataclass
class EventTrigger(Codable):
    condition: DerivedFeature
    event: StreamDataSource
    payload: set[Feature]

    async def check_pandas(self, df: pd.DataFrame, result: RequestResult) -> None:
        from aligned.data_source.stream_data_source import SinkableDataSource
        from aligned.local.job import LiteralRetrivalJob

        if not isinstance(self.event, SinkableDataSource):
            logger.info(f'Event: {self.event.topic_name} is not sinkable, will return')
            return

        logger.info(f'Checking for event: {self.event.topic_name}')

        mask = await self.condition.transformation.transform_pandas(df)

        if mask.any():
            trigger_result = RequestResult(result.entities, self.payload, None)
            features = {entity.name for entity in result.entities}.union(
                {feature.name for feature in self.payload}
            )
            events = df[list(features)].loc[mask]
            logger.info(f'Sending {events.shape[0]} events: {self.event.topic_name}')
            await self.event.write_to_stream(LiteralRetrivalJob(events, trigger_result))

    async def check_polars(self, df: pl.LazyFrame, result: RequestResult) -> None:
        from aligned.data_source.stream_data_source import SinkableDataSource
        from aligned.local.job import LiteralRetrivalJob

        if not isinstance(self.event, SinkableDataSource):
            logger.info(f'Event: {self.event.topic_name} is not sinkable, will return')
            return

        logger.info(f'Checking for event: {self.event.topic_name}')

        mask: pl.LazyFrame = await self.condition.transformation.transform_polars(df, self.condition.name)
        mask = mask.filter(pl.col(self.condition.name))

        triggers = mask.collect()

        if triggers.shape[0] > 0:
            trigger_result = RequestResult(result.entities, self.payload, None)
            features = {entity.name for entity in result.entities}.union(
                {feature.name for feature in self.payload}
            )
            events = mask.lazy().select(features)
            logger.info(f'Sending {triggers.shape[0]} events: {self.event.topic_name}')
            await self.event.write_to_stream(LiteralRetrivalJob(events, trigger_result))

    def __hash__(self) -> int:
        return self.event.topic_name.__hash__()
