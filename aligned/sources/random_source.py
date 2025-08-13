from __future__ import annotations

import polars as pl

import logging
from typing import Any, Literal
from datetime import timedelta, timezone, datetime

from aligned.data_file import DataFileReference, upsert_on_column
from aligned.retrieval_job import RetrievalJob

from aligned.feature_source import WritableFeatureSource
from aligned.schemas.feature import Feature, FeatureLocation, FeatureType
from aligned.request.retrieval_request import RetrievalRequest
from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    BatchDataSource,
    CustomMethodDataSource,
)

logger = logging.getLogger(__name__)


def random_values_for(
    feature: Feature, size: int, seed: int | None = None
) -> pl.Series:
    from aligned.schemas.constraints import (
        InDomain,
        LowerBound,
        LowerBoundInclusive,
        Unique,
        UpperBound,
        UpperBoundInclusive,
        Optional,
        ListConstraint,
    )
    import numpy as np

    if seed:
        np.random.seed(seed)

    dtype = feature.dtype

    choices: list[Any] | None = None
    max_value: float | None = None
    min_value: float | None = None

    is_optional = False
    is_unique = False

    for constraints in feature.constraints or set():
        if isinstance(constraints, InDomain):
            choices = constraints.values
        elif isinstance(constraints, LowerBound):
            min_value = constraints.value
        elif isinstance(constraints, LowerBoundInclusive):
            min_value = constraints.value
        elif isinstance(constraints, UpperBound):
            max_value = constraints.value
        elif isinstance(constraints, UpperBoundInclusive):
            max_value = constraints.value
        elif isinstance(constraints, Unique):
            is_unique = True
        elif isinstance(constraints, Optional):
            is_optional = True

    if dtype == FeatureType.boolean():
        values = np.random.choice([True, False], size=size)
    elif dtype.is_numeric:
        if is_unique:
            values = np.arange(0, size, dtype=dtype.pandas_type)
        else:
            values = np.random.random(size)

        if max_value is None and dtype.name.startswith("uint"):
            bits = dtype.name.lstrip("uint")
            if bits.isdigit():
                max_value = 2 ** int(bits) - 1
                min_value = 0
        elif max_value is None and dtype.name.startswith("int"):
            bits = dtype.name.lstrip("int")
            if bits.isdigit():
                value_range = 2 ** int(bits) / 2
                max_value = value_range - 1
                min_value = -value_range

        if max_value and min_value:
            values = values * (max_value - min_value) + min_value
        elif max_value is not None:
            values = values * max_value
        elif min_value is not None:
            values = values * 1000 + min_value

        if "float" not in dtype.name:
            values = np.round(values)

    elif dtype.is_datetime:
        values = [
            datetime.now(tz=timezone.utc) - np.random.random() * timedelta(days=365)
            for _ in range(size)
        ]
    elif dtype.is_array:
        subtype = dtype.array_subtype()
        _sub_constraints: list[ListConstraint] = [
            constraint
            for constraint in feature.constraints or set()
            if isinstance(constraint, ListConstraint)
        ]
        sub_constraints = None
        if _sub_constraints:
            sub_constraints = set(_sub_constraints[0].constraints)

        if subtype is None:
            values = np.random.random((size, 4))
        else:
            values = [
                random_values_for(
                    Feature("dd", dtype=subtype, constraints=sub_constraints), 4
                )
                for _ in range(size)
            ]
    elif dtype.is_embedding:
        embedding_size = dtype.embedding_size() or 10
        values = np.random.random((size, embedding_size))
    else:
        if choices:
            values = np.random.choice(choices, size=size)
        else:
            values = np.random.choice(list("abcde"), size=size)

    pl_vals = pl.Series(values=values)
    if is_optional:
        pl_vals = pl_vals.set(
            pl.Series(values=np.random.random(size) > 0.5), value=None
        )

    return pl_vals


async def data_for_request(
    request: RetrievalRequest,
    size: int,
    seed: int | None = None,
    values: pl.DataFrame | None = None,
) -> pl.DataFrame:
    needed_features = request.features.union(request.entities)
    if request.event_timestamp:
        needed_features.add(request.event_timestamp.as_feature())

    if values is not None:
        needed_features = {
            feat for feat in needed_features if feat.name not in values.columns
        }

    schema = {feature.name: feature.dtype.polars_type for feature in needed_features}
    exprs = {}

    for feature in sorted(needed_features, key=lambda f: f.name):
        logger.info(f"Generating data for {feature.name}")
        exprs[feature.name] = random_values_for(feature, size, seed)

    random_features = pl.DataFrame(exprs, schema=schema)

    if values is not None:
        join_columns = list(set(schema.keys()) - set(values.columns))

        if size == 1:
            random_features = pl.concat(
                [random_features] * values.height, how="vertical"
            )

        raw_data = values.hstack(random_features.select(pl.col(join_columns)))
    else:
        raw_data = random_features

    return (
        await RetrievalJob.from_convertable(raw_data, request)
        .derive_features()
        .to_polars()
    )


FillMode = Literal["duplicate", "random_samples"]


class RandomDataSource(
    CodableBatchDataSource, DataFileReference, WritableFeatureSource
):
    """
    The DummyDataBatchSource is a data source that generates random data for a given request.
    This can be useful for testing and development purposes.

    It will use the data types and constraints defined on a feature to generate the data.

    ```python
    from aligned import feature_view, Int64, String, DummyDataBatchSource

    @feature_view(
        source=RandomDataSource(),
    )
    class MyView:
        passenger_id = Int64().as_entity()
        survived = Bool()
        age = Float().lower_bound(0).upper_bound(100)
        name = String()
        sex = String().accepted_values(["male", "female"])
    ```
    """

    default_data_size: int
    seed: int | None

    fill_mode: FillMode

    type_name: str = "dummy_data"

    def __init__(
        self,
        default_data_size: int = 10_000,
        seed: int | None = None,
        partial_data: pl.DataFrame | None = None,
        fill_mode: FillMode = "duplicate",
    ):
        self.default_data_size = default_data_size
        self.seed = seed
        if partial_data is None:
            self.partial_data = pl.DataFrame()
        else:
            self.partial_data = partial_data
        self.fill_mode = fill_mode

    def job_group_key(self) -> str:
        return self.type_name

    async def insert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        values = await job.to_polars()
        data = self.partial_data
        if not data.is_empty():
            self.partial_data = data.vstack(values.select(data.columns))
        else:
            self.partial_data = values

    async def upsert(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        values = await job.to_lazy_polars()

        self.partial_data = upsert_on_column(
            sorted(request.entity_names),
            new_data=values,
            existing_data=self.partial_data.lazy(),
        ).collect()

    async def overwrite(self, job: RetrievalJob, request: RetrievalRequest) -> None:
        self.partial_data = await job.to_polars()

    async def write_polars(self, df: pl.LazyFrame) -> None:
        self.partial_data = df.collect()

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls: type["RandomDataSource"],
        facts: RetrievalJob,
        requests: list[tuple["RandomDataSource", RetrievalRequest]],
    ) -> RetrievalJob:
        from aligned.local.job import FileFactualJob

        sources = {
            source.job_group_key()
            for source, _ in requests
            if isinstance(source, BatchDataSource)
        }
        if len(sources) != 1:
            raise NotImplementedError(
                f"Type: {cls} have not implemented how to load fact data with multiple sources."
            )

        source, _ = requests[0]

        request = RetrievalRequest.unsafe_combine([request for _, request in requests])
        join_entities = request.entity_names.intersection(
            facts.request_result.all_returned_columns
        )

        async def random_features_for(
            facts: RetrievalJob, request: RetrievalRequest
        ) -> pl.LazyFrame:
            if not join_entities or source.partial_data.is_empty():
                df = await facts.to_polars()
                if source.fill_mode == "duplicate":
                    random = (await data_for_request(request, 1, values=df)).lazy()
                else:
                    random = (
                        await data_for_request(request, df.height, values=df)
                    ).lazy()

                return random

            join_columns = set(request.all_returned_columns) - set(
                source.partial_data.columns
            )
            if not join_columns:
                return source.partial_data.lazy()

            partial_data = source.partial_data
            if source.fill_mode == "duplicate":
                return (await data_for_request(request, 1, values=partial_data)).lazy()
            else:
                return (
                    await data_for_request(
                        request, partial_data.height, values=partial_data
                    )
                ).lazy()

        if not join_entities:
            return CustomMethodDataSource.from_methods(
                features_for=random_features_for,
            ).features_for(facts, request)

        return FileFactualJob(
            CustomMethodDataSource.from_methods(
                features_for=random_features_for,
            ).features_for(facts, request),
            [request for _, request in requests],
            facts,
        )

    def all_data(
        self, request: RetrievalRequest, limit: int | None = None
    ) -> RetrievalJob:
        from aligned import CustomMethodDataSource

        async def all_data(
            request: RetrievalRequest, limit: int | None = None
        ) -> pl.LazyFrame:
            full_df = self.partial_data

            if full_df.is_empty():
                return (
                    await data_for_request(request, limit or self.default_data_size)
                ).lazy()

            if limit:
                full_df = self.partial_data.head(limit)

            join_columns = set(request.all_returned_columns) - set(full_df.columns)
            if not join_columns:
                return full_df.lazy()

            random_df = (
                await data_for_request(request, full_df.height, values=full_df)
            ).lazy()
            return random_df

        return CustomMethodDataSource.from_methods(all_data=all_data).all_data(
            request, limit
        )

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        from aligned import CustomMethodDataSource

        async def between_date(
            request: RetrievalRequest, start_date: datetime, end_date: datetime
        ) -> pl.LazyFrame:
            return (await data_for_request(request, self.default_data_size)).lazy()

        return CustomMethodDataSource.from_methods(
            all_between_dates=between_date
        ).all_between_dates(request, start_date, end_date)

    async def schema(self) -> dict[str, FeatureType]:
        return {}

    def depends_on(self) -> set[FeatureLocation]:
        return set()

    @staticmethod
    def with_values(
        values: dict[str, object], seed: int | None = None
    ) -> "RandomDataSource":
        return RandomDataSource(seed=seed, partial_data=pl.DataFrame(values))
