from __future__ import annotations

import logging
from typing import Callable

import polars as pl

from aligned.lazy_imports import pandas as pd, pandera as pa
from aligned.schemas.constraints import Constraint, Optional
from aligned.schemas.feature import Feature, FeatureType
from aligned.validation.interface import Validator

logger = logging.getLogger(__name__)


class PanderaValidator(Validator):

    check_map: dict[str, Callable[[Constraint], pa.Check]] = {
        'lower_bound': lambda constraint: pa.Check.greater_than(constraint.value),  # type: ignore
        'lower_bound_inc': lambda constraint: pa.Check.greater_than_or_equal_to(constraint.value),  # type: ignore
        'upper_bound': lambda constraint: pa.Check.less_than(constraint.value),  # type: ignore
        'upper_bound_inc': lambda constraint: pa.Check.less_than_or_equal_to(constraint.value),  # type: ignore
        'in_domain': lambda domain: pa.Check.isin(domain.values),  # type: ignore
        'min_length': lambda constraint: pa.Check.str_length(min_value=constraint.value),  # type: ignore
        'max_length': lambda constraint: pa.Check.str_length(max_value=constraint.value),  # type: ignore
        'regex': lambda constraint: pa.Check.str_matches(constraint.value),  # type: ignore
        'ends_with': lambda constraint: pa.Check.str_endswith(constraint.value),  # type: ignore
        'starts_with': lambda constraint: pa.Check.str_startswith(constraint.value),  # type: ignore
    }

    datatype_check = {
        FeatureType.string(),
        FeatureType.uuid(),
        FeatureType.date(),
        FeatureType.int32(),
        FeatureType.int64(),
    }

    def _column_for(self, feature: Feature) -> pa.Column:

        if feature.constraints is None:
            return pa.Column(
                feature.dtype.pandas_type if feature.dtype in self.datatype_check else None,
                nullable=False,
                coerce=True,
            )

        is_nullable = Optional() in feature.constraints

        checks = [
            self.check_map[constraint.name](constraint)
            for constraint in feature.constraints
            if constraint.name in self.check_map
        ]

        return pa.Column(
            dtype=feature.dtype.pandas_type if feature.dtype in self.datatype_check else None,
            checks=checks,
            nullable=is_nullable,
            required=not is_nullable,
        )

    def _build_schema(self, features: list[Feature]) -> pa.DataFrameSchema:
        return pa.DataFrameSchema(
            columns={feature.name: self._column_for(feature) for feature in features}, drop_invalid_rows=True
        )

    def validate_pandas(self, features: list[Feature], df: pd.DataFrame) -> pd.DataFrame:
        schema = self._build_schema(features)
        return schema.validate(df, lazy=True)

    def validate_polars(self, features: list[Feature], df: pl.LazyFrame) -> pl.LazyFrame:
        input_df = df.collect().to_pandas()
        return pl.from_pandas(self.validate_pandas(features, input_df)).lazy()
