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
    def max_confidence() -> ActiveLearningMetric:
        def metric_selection(model: Model) -> pl.Expr:
            view = model.predictions_view

            if view.classification_targets and len(view.classification_targets) > 0:
                confidence = [prob.confidence.name for prob in view.classification_targets if prob.confidence]
                return pl.concat_list(confidence).arr.max()

            if view.regression_targets and len(view.regression_targets) > 0:
                confidence = [prob.confidence.name for prob in view.regression_targets if prob.confidence]
                return pl.concat_list(confidence).arr.max()

            return pl.lit(1)

        return ActivLearningPolarsExprMetric(metric_selection)


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
