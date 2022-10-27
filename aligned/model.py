import logging
from datetime import datetime, timedelta
from typing import Callable

from pandas import DataFrame

from aligned.compiler.feature_factory import FeatureFactory
from aligned.entity_data_source import EntityDataSource
from aligned.request.retrival_request import FeatureRequest, RetrivalRequest

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


class ModelService:
    feature_refs: set[str]
    target_refs: set[str] | None

    _name: str | None
    entity_source: SqlEntityDataSource | None

    @property
    def name(self) -> str:
        if not self._name:
            raise ValueError('Model name is not set')
        return self._name

    def __init__(
        self,
        features: list[FeatureRequest],
        targets: list[RetrivalRequest] | list[FeatureFactory] | None = None,
        name: str | None = None,
        entity_source: SqlEntityDataSource | None = None,
    ) -> None:
        self._name = name
        self.entity_source = entity_source
        self.feature_refs = set()
        self.target_refs = set()
        for request in features:
            self.feature_refs.update({f'{request.name}:{feature}' for feature in request.features_to_include})
        for request in targets or []:
            if isinstance(targets, FeatureFactory):
                self.target_refs = {f'{target}' for target in targets}
            else:
                self.target_refs.update(
                    {f'{request.feature_view_name}:{feature}' for feature in request.all_feature_names}
                )
