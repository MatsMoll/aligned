import logging
from typing import Callable

import pandas as pd
from pandera import Check, Column, DataFrameSchema  # type: ignore[attr-defined]

from aligned.schemas.constraints import Constraint, Required
from aligned.schemas.feature import Feature
from aligned.validation.interface import Validator

logger = logging.getLogger(__name__)


class PanderaValidator(Validator):

    check_map: dict[str, Callable[[Constraint], Check]] = {
        'lower_bound': lambda constraint: Check.greater_than(constraint.value),
        'lower_bound_inc': lambda constraint: Check.greater_than_or_equal_to(constraint.value),
        'upper_bound': lambda constraint: Check.less_than(constraint.value),
        'upper_bound_inc': lambda constraint: Check.less_than_or_equal_to(constraint.value),
        'in_domain': lambda domain: Check.isin(domain.values),
    }

    def _column_for(self, feature: Feature) -> Column:
        if feature.constraints is None:
            return Column(feature.dtype.pandas_type, nullable=True)

        is_nullable = Required() not in feature.constraints

        return Column(
            feature.dtype.pandas_type,
            checks=[
                self.check_map[constraint.name](constraint)
                for constraint in feature.constraints
                if constraint.name in self.check_map
            ],
            nullable=is_nullable,
            required=not is_nullable,
        )

    def _build_schema(self, features: list[Feature]) -> DataFrameSchema:
        return DataFrameSchema(columns={feature.name: self._column_for(feature) for feature in features})

    async def validate_pandas(self, features: list[Feature], df: pd.DataFrame) -> pd.DataFrame:
        from pandera.errors import SchemaError

        schema = self._build_schema(features)
        try:
            return schema.validate(df)
        except SchemaError as error:
            # Will only return one error at a time, so will remove
            # errors and then run it recrusive

            if error.failure_cases.shape[0] == df.shape[0]:
                raise ValueError('Validation is removing all the data.')

            return await self.validate_pandas(
                features, df.loc[df.index.delete(error.failure_cases['index'])].reset_index()
            )
