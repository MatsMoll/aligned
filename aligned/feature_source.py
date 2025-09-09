from __future__ import annotations

from typing import TYPE_CHECKING

from aligned.request.retrieval_request import FeatureRequest, RetrievalRequest
from aligned.retrieval_job import RetrievalJob
from aligned.schemas.feature import FeatureLocation, Feature
from aligned.schemas.transformation import Expression

if TYPE_CHECKING:
    from datetime import datetime


class FeatureSourceFactory:
    def feature_source(self) -> FeatureSource:
        raise NotImplementedError()


class FeatureSource:
    def features_for(
        self, facts: RetrievalJob, request: FeatureRequest
    ) -> RetrievalJob:
        raise NotImplementedError()

    async def freshness_for(
        self, locations: dict[FeatureLocation, Feature]
    ) -> dict[FeatureLocation, datetime | None]:
        raise NotImplementedError()


class WritableFeatureSource:
    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        raise NotImplementedError(f"Append is not implemented for {type(self)}.")

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        raise NotImplementedError(f"Upsert write is not implemented for {type(self)}.")

    async def overwrite(
        self,
        job: RetrievalJob,
        request: RetrievalRequest,
        predicate: Expression | None = None,
    ) -> None:
        raise NotImplementedError(
            f"Overwrite write is not implemented for {type(self)}."
        )


class RangeFeatureSource:
    def all_for(
        self, request: FeatureRequest, limit: int | None = None
    ) -> RetrievalJob:
        raise NotImplementedError()

    def all_between(
        self, start_date: datetime, end_date: datetime, request: FeatureRequest
    ) -> RetrievalJob:
        raise NotImplementedError()
