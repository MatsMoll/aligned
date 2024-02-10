import logging
from dataclasses import dataclass

import pandas as pd
import polars as pl

from aligned.active_learning.selection import ActiveLearningMetric, ActiveLearningSelection
from aligned.active_learning.write_policy import ActiveLearningWritePolicy
from aligned.retrival_job import RetrivalJob
from aligned.schemas.model import Model

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningJob(RetrivalJob):

    job: RetrivalJob
    model: Model
    metric: ActiveLearningMetric
    selection: ActiveLearningSelection
    write_policy: ActiveLearningWritePolicy

    async def to_lazy_polars(self) -> pl.LazyFrame:
        if not self.model.predictions_view.classification_targets:
            logger.info('Found no target. Therefore, no data will be written to an active learning dataset.')
            return await self.job.to_lazy_polars()

        data = await self.job.to_lazy_polars()
        active_learning_set = self.selection.select(self.model, data, self.metric)
        await self.write_policy.write(active_learning_set, self.model)
        return data

    async def to_pandas(self) -> pd.DataFrame:
        raise NotImplementedError()
