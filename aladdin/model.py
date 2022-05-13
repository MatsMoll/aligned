from datetime import datetime
from pandas import DataFrame

class FactDataSource:

    def all_in_range(self, start_date: datetime, end_date: datetime) -> DataFrame:
        pass

    def last(self, days: int, hours: int, seconds: int) -> DataFrame:
        pass


class ModelFeatures:
    
    name: str
    feature_refs: list[str]
    fact_source: FactDataSource | None
