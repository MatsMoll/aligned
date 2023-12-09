import logging
from typing import Callable

import pandas as pd
import polars as pl
from pandera import Check, Column, DataFrameSchema  # type: ignore[attr-defined]

from aligned.schemas.constraints import Constraint, Optional
from aligned.schemas.feature import Feature, FeatureType
from aligned.validation.interface import Validator

logger = logging.getLogger(__name__)


class PanderaValidator(Validator):

    check_map: dict[str, Callable[[Constraint], Check]] = {
        'lower_bound': lambda constraint: Check.greater_than(constraint.value),
        'lower_bound_inc': lambda constraint: Check.greater_than_or_equal_to(constraint.value),
        'upper_bound': lambda constraint: Check.less_than(constraint.value),
        'upper_bound_inc': lambda constraint: Check.less_than_or_equal_to(constraint.value),
        'in_domain': lambda domain: Check.isin(domain.values),
        'min_length': lambda constraint: Check.str_length(min_value=constraint.value),
        'max_length': lambda constraint: Check.str_length(max_value=constraint.value),
        'regex': lambda constraint: Check.str_matches(constraint.value),
        'ends_with': lambda constraint: Check.str_endswith(constraint.value),
        'starts_with': lambda constraint: Check.str_startswith(constraint.value),
    }

    datatype_check = {
        # FeatureType.bool(),
        FeatureType.string(),
        FeatureType.uuid(),
        FeatureType.date(),
        FeatureType.int32(),
        FeatureType.int64(),
    }

    def _column_for(self, feature: Feature) -> Column:

        if feature.constraints is None:
            return Column(
                feature.dtype.pandas_type if feature.dtype in self.datatype_check else None,
                nullable=True,
                coerce=True,
            )

        is_nullable = Optional() in feature.constraints

        checks = [
            self.check_map[constraint.name](constraint)
            for constraint in feature.constraints
            if constraint.name in self.check_map
        ]

        return Column(
            dtype=feature.dtype.pandas_type if feature.dtype in self.datatype_check else None,
            checks=checks,
            nullable=is_nullable,
            required=not is_nullable,
        )

    def _build_schema(self, features: list[Feature]) -> DataFrameSchema:
        return DataFrameSchema(
            columns={feature.name: self._column_for(feature) for feature in features}, drop_invalid_rows=True
        )

    def validate_pandas(self, features: list[Feature], df: pd.DataFrame) -> pd.DataFrame:
        schema = self._build_schema(features)
        return schema.validate(df, lazy=True)

    def validate_polars(self, features: list[Feature], df: pl.LazyFrame) -> pl.LazyFrame:
        input_df = df.collect().to_pandas()
        return pl.from_pandas(self.validate_pandas(features, input_df)).lazy()
