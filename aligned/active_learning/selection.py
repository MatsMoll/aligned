from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl

from aligned.schemas.model import Model


@dataclass
class ActiveLearningMetric:
    def metric(self, model: Model) -> pl.Expr:
        raise NotImplementedError()

    @staticmethod
    def max_probability() -> ActiveLearningMetric:
        return ActivLearningPolarsExprMetric(
            lambda model: pl.concat_list(
                [prob.feature.name for prob in model.predictions_view.probabilities]
            ).arr.max()
        )


@dataclass
class ActivLearningPolarsExprMetric(ActiveLearningMetric):

    factory: Callable[[Model], pl.Expr]

    def metric(self, model: Model) -> pl.Expr:
        return self.factory(model)


class ActiveLearningSelection:
    def select(self, model: Model, data: pl.LazyFrame, metric: ActiveLearningMetric) -> pl.LazyFrame:
        raise NotImplementedError()

    @staticmethod
    def under_threshold(threshold: float) -> ActiveLearningSelection:
        return ActivLearningPolarsExprSelection(lambda model, metric: metric.metric(model) < threshold)


@dataclass
class ActivLearningPolarsExprSelection(ActiveLearningSelection):

    factory: Callable[[Model, ActiveLearningMetric], pl.Expr]

    def select(self, model: Model, data: pl.LazyFrame, metric: ActiveLearningMetric) -> pl.LazyFrame:
        return data.filter(self.factory(model, metric))
