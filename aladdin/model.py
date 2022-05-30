from datetime import datetime, timedelta
from typing import Callable
from pandas import DataFrame

from aladdin.feature_types import FeatureReferancable
from aladdin.entity_data_source import EntityDataSource
from aladdin.request.retrival_request import RetrivalRequest


class SqlEntityDataSource(EntityDataSource):
    
    url: str
    timestamp_column: str

    def __init__(self, sql: Callable[[str], str], url: str, timestamp_column: str) -> None:
        self.sql = sql
        self.url = url
        self.timestamp_column = timestamp_column

    async def all_in_range(self, start_date: datetime, end_date: datetime) -> DataFrame:
        query = self.sql(f"{self.timestamp_column} BETWEEN (:start_date) AND (:end_date)")
        from databases import Database
        try:
            async with Database(self.url) as db:
                records = await db.fetch_all(query=query, values={
                    "start_date": start_date,
                    "end_date": end_date
                })
        except Exception as e:
            print(query)
            print(e)
        return DataFrame.from_records([dict(record) for record in records])

    async def last(self, days: int, hours: int, seconds: int) -> DataFrame:
        now = datetime.utcnow()
        return await self.all_in_range(
            now - timedelta(days=days, hours=hours, seconds=seconds),
            now
        )

class ModelFeatures:
    feature_refs: set[str]
    target_refs: set[str] | None
    name: str | None
    entity_source: SqlEntityDataSource | None


    def __init__(self, features: list[RetrivalRequest], targets: list[RetrivalRequest] | list[FeatureReferancable] | None = None, name: str | None = None, entity_source: SqlEntityDataSource | None = None) -> None:
        self.name = name
        self.entity_source = entity_source
        self.feature_refs = set()
        self.target_refs = set()
        for request in features:
            self.feature_refs.update({
                f"{request.feature_view_name}:{feature}" 
                for feature in request.all_feature_names
            })
        for request in (targets or []):
            if isinstance(targets, FeatureReferancable):
                self.target_refs = [f"{target}" for target in targets]
            else:
                self.target_refs.update({
                    f"{request.feature_view_name}:{feature}" 
                    for feature in request.all_feature_names
                })