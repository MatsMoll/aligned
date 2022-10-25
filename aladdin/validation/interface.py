import pandas as pd

from aladdin.schemas.feature import Feature


class Validator:
    async def validate_pandas(self, features: list[Feature], df: pd.DataFrame) -> pd.DataFrame:
        pass
