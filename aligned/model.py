import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from pandas import DataFrame

from aligned.compiler.feature_factory import FeatureFactory
from aligned.entity_data_source import EntityDataSource
from aligned.request.retrival_request import FeatureRequest
from aligned.schemas.feature import FeatureReferance
from aligned.schemas.model import Model as ModelSchema

logger = logging.getLogger(__name__)


class SqlEntityDataSource(EntityDataSource):

    url: str
    timestamp_column: str

    def __init__(self, sql: Callable[[str], str], url: str, timestamp_column: str) -> None:
        self.sql = sql
        self.url = url
        self.timestamp_column = timestamp_column

    async def all_in_range(self, start_date: datetime, end_date: datetime) -> DataFrame:
        import os

        query = self.sql(f'{self.timestamp_column} BETWEEN (:start_date) AND (:end_date)')
        from databases import Database

        try:
            async with Database(os.environ[self.url]) as db:
                records = await db.fetch_all(
                    query=query, values={'start_date': start_date, 'end_date': end_date}
                )
        except Exception as error:
            logger.info(query)
            logger.error(error)
            raise error

        return DataFrame.from_records([dict(record) for record in records])

    async def last(self, days: int, hours: int, seconds: int) -> DataFrame:
        now = datetime.utcnow()
        return await self.all_in_range(now - timedelta(days=days, hours=hours, seconds=seconds), now)


@dataclass
class Model:
    features: set[FeatureRequest]
    target: list[FeatureRequest] | FeatureFactory | None = field(default=None)
    name: str | None = field(default=None)

    def schema(self) -> ModelSchema:
        if not self.name:
            raise ValueError(
                'Missing name for model. You man need to set it manually using the `name` property.'
            )
        features: set[FeatureReferance] = set()
        targets: set[FeatureReferance] = set()

        for request in self.features:
            features.update(
                {
                    FeatureReferance(feature.name, request.name, feature.dtype)
                    for feature in request.request_result.features
                }
            )

        if self.target:
            if isinstance(self.target, FeatureFactory):
                targets.add(self.target.feature_referance())
            else:
                for request in self.target:
                    targets.update(
                        {
                            FeatureReferance(feature.name, request.name, feature.dtype)
                            for feature in request.request_result.features
                        }
                    )

        return ModelSchema(
            name=self.name,
            features=features,
            targets=targets if targets else None,
        )
