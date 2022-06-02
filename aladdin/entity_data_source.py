from datetime import datetime

from pandas import DataFrame


class EntityDataSource:
    async def all_in_range(self, start_date: datetime, end_date: datetime) -> DataFrame:
        pass

    async def last(self, days: int, hours: int, seconds: int) -> DataFrame:
        pass
