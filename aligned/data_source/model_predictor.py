from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

from aligned.feature_store import ModelFeatureStore
from aligned.request.retrival_request import RetrivalRequest
from aligned.schemas.feature import FeatureLocation, FeatureType
from aligned.schemas.model import Model
from aligned.retrival_job import RetrivalJob


@dataclass
class PredictModelSource:

    store: ModelFeatureStore
    type_name: str = 'pred_model_source'

    @property
    def model(self) -> Model:
        return self.store.model

    def job_group_key(self) -> str:
        loc = FeatureLocation.model(self.model.name).identifier
        return f"{loc}_pred"

    def location_id(self) -> set[FeatureLocation]:
        return {FeatureLocation.model(self.model.name)}

    async def schema(self) -> dict[str, FeatureType]:
        if self.model.predictions_view.source:
            return await self.model.predictions_view.source.schema()
        return {}

    def all_data(self, request: RetrivalRequest, limit: int | None = None) -> RetrivalJob:
        reqs = self.store.request()
        if len(reqs.needed_requests) != 1:
            raise NotImplementedError(
                f'Type: {type(self)} have not implemented how to load fact data with multiple sources.'
            )

        location = reqs.needed_requests[0].location
        if location.location != 'feature_view':
            raise NotImplementedError(
                f'Type: {type(self)} have not implemented how to load fact data with multiple sources.'
            )

        entities = self.store.store.feature_view(location.name).all_columns(limit=limit)
        return self.store.predict_over(entities).with_request([request])

    def all_between_dates(
        self, request: RetrivalRequest, start_date: datetime, end_date: datetime
    ) -> RetrivalJob:
        reqs = self.store.request()
        if len(reqs.needed_requests) != 1:
            raise NotImplementedError(
                f'Type: {type(self)} have not implemented how to load fact data with multiple sources.'
            )

        location = reqs.needed_requests[0].location
        if location.location != 'feature_view':
            raise NotImplementedError(
                f'Type: {type(self)} have not implemented how to load fact data with multiple sources.'
            )

        entities = self.store.store.feature_view(location.name).between_dates(start_date, end_date)
        return self.store.predict_over(entities).with_request([request])

    def features_for(self, facts: RetrivalJob, request: RetrivalRequest) -> RetrivalJob:
        return self.store.predict_over(facts).with_request([request])

    @classmethod
    def multi_source_features_for(
        cls, facts: RetrivalJob, requests: list[tuple[PredictModelSource, RetrivalRequest]]
    ) -> RetrivalJob:

        if len(requests) != 1:
            raise NotImplementedError(
                f'Type: {cls} have not implemented how to load fact data with multiple sources.'
            )

        source, _ = requests[0]
        return source.features_for(facts, requests[0][1])

    def depends_on(self) -> set[FeatureLocation]:
        return {FeatureLocation.model(self.model.name)}
