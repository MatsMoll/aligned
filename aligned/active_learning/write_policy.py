from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl

from aligned.schemas.model import Model

logger = logging.getLogger(__name__)


class ActiveLearningWritePolicy:
    async def write(self, data: pl.LazyFrame, model: Model):
        raise NotImplementedError()

    @staticmethod
    def sample_size(write_size: int, ideal_size: int) -> ActiveLearningWritePolicy:
        return ActiveLearningSampleSizePolicy(write_size, ideal_size)


@dataclass
class ActiveLearningSampleSizePolicy(ActiveLearningWritePolicy):

    write_size: int
    ideal_size: int

    dataset_folder_name: str = field(default='active_learning')
    dataset_file_name: str = field(default='data.csv')

    unsaved_size: float = field(default=0)
    write_timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    current_frame: pl.DataFrame = field(default_factory=lambda: pl.DataFrame())

    async def write(self, data: pl.LazyFrame, model: Model):

        if not model.dataset_store:
            logger.info(
                'Found no dataset folder. Therefore, no data will be written to an active learning dataset.'
            )
            return

        collected_data = data.collect()

        if self.current_frame.shape[0] == 0:
            self.current_frame = collected_data
        else:
            self.current_frame = self.current_frame.extend(collected_data)

        self.unsaved_size += collected_data.shape[0]

        if self.unsaved_size >= self.write_size or self.current_frame.shape[0] >= self.ideal_size:
            dataset_subfolder = Path(self.dataset_folder_name) / str(self.write_timestamp)
            logger.info(f'Writing active learning data to {dataset_subfolder}')

            dataset = model.dataset_store.file_at(dataset_subfolder / self.dataset_file_name)
            await dataset.write(self.current_frame.write_csv().encode('utf-8'))
            self.unsaved_size = 0

        if self.current_frame.shape[0] >= self.ideal_size:
            self.write_timestamp = datetime.utcnow().timestamp()
            self.current_frame = pl.DataFrame()
