import numpy as np
import polars as pl
import pytest
from redis.asyncio.client import Pipeline  # type: ignore

from aligned.redis.job import FactualRedisJob
from aligned.request.retrieval_request import RetrievalRequest
from aligned.retrieval_job import RetrievalJob
from aligned.schemas.date_formatter import DateFormatter
from aligned.schemas.feature import Feature, FeatureLocation, FeatureType
from aligned.sources.redis import RedisConfig, RedisSource


@pytest.fixture
def retrieval_request() -> RetrievalRequest:
    return RetrievalRequest(
        name="test",
        location=FeatureLocation.feature_view("test"),
        entities={
            Feature(name="id_int", dtype=FeatureType.int32()),
            Feature(name="id_str", dtype=FeatureType.string()),
        },
        features={
            Feature(name="x", dtype=FeatureType.int32()),
        },
        derived_features=set(),
        event_timestamp=None,
    )


@pytest.mark.asyncio
async def test_factual_redis_job(mocker, retrieval_request) -> None:  # type: ignore[no-untyped-def]
    values = ["20", "44"]

    redis_mock = mocker.patch.object(Pipeline, "execute", return_value=values)

    facts = RetrievalJob.from_dict(
        data={"id_int": [1.0, 2.0, None, 4.0], "id_str": ["a", "b", "c", None]},
        request=retrieval_request,
    )
    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrieval_request],
        facts=facts,
        formatter=DateFormatter.iso_8601(),
    )

    result = await job.to_pandas()
    redis_mock.assert_called_once()
    x_result = [int(value) for value in values] + [0, 0]
    assert np.all(result["x"].fillna(0).values == x_result), f"Got {result}"  # type: ignore


@pytest.mark.asyncio
async def test_factual_redis_job_int_as_str(mocker, retrieval_request) -> None:  # type: ignore[no-untyped-def]
    values = ["20", "44"]

    redis_mock = mocker.patch.object(Pipeline, "execute", return_value=values)

    facts = RetrievalJob.from_dict(
        data={"id_int": ["1", "2", None, "4"], "id_str": ["a", "b", "c", None]},
        request=retrieval_request,
    )

    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrieval_request],
        facts=facts,
        formatter=DateFormatter.iso_8601(),
    )

    result = await job.to_pandas()
    redis_mock.assert_called_once()
    x_result = [int(value) for value in values] + [0, 0]
    assert np.all(result["x"].fillna(0).values == x_result)  # type: ignore


@pytest.mark.asyncio
async def test_nan_entities_job(mocker, retrieval_request) -> None:  # type: ignore[no-untyped-def]
    values = ["20", "44"]

    redis_mock = mocker.patch.object(Pipeline, "execute", return_value=values)

    facts = RetrievalJob.from_dict(
        data={"id_int": [None, "4"], "id_str": ["c", None]},
        request=retrieval_request,
    )
    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrieval_request],
        facts=facts,
        formatter=DateFormatter.iso_8601(),
    )

    _ = await job.to_pandas()
    redis_mock.assert_not_called()


@pytest.mark.asyncio
async def test_no_entities_job(mocker, retrieval_request) -> None:  # type: ignore[no-untyped-def]
    values = ["20", "44"]

    redis_mock = mocker.patch.object(Pipeline, "execute", return_value=values)

    facts = RetrievalJob.from_dict(
        data={"id_int": [], "id_str": []},
        request=retrieval_request,
    )
    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrieval_request],
        facts=facts,
        formatter=DateFormatter.iso_8601(),
    )

    _ = await job.to_pandas()
    redis_mock.assert_not_called()


@pytest.mark.asyncio
async def test_factual_redis_job_int_entity(mocker) -> None:  # type: ignore[no-untyped-def]
    retrieval_request = RetrievalRequest(
        name="test",
        location=FeatureLocation.feature_view("test"),
        entities={Feature(name="id_int", dtype=FeatureType.int32())},
        features={
            Feature(name="x", dtype=FeatureType.int32()),
        },
        derived_features=set(),
        event_timestamp=None,
    )

    values = ["20", "44", "55"]

    redis_mock = mocker.patch.object(Pipeline, "execute", return_value=values)

    facts = RetrievalJob.from_dict(
        data={"id_int": [1.0, 2.0, 4.0, None]},
        request=retrieval_request,
    )
    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrieval_request],
        facts=facts,
        formatter=DateFormatter.iso_8601(),
    )

    result = await job.to_pandas()
    redis_mock.assert_called_once()
    x_result = [int(value) for value in values] + [0]
    assert np.all(result["x"].fillna(0).values == x_result)  # type: ignore


@pytest.mark.asyncio
async def test_write_job(mocker, retrieval_request: RetrievalRequest) -> None:  # type: ignore[no-untyped-def]
    import fakeredis.aioredis

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

    _ = mocker.patch.object(RedisConfig, "redis", return_value=redis)

    insert_facts = RetrievalJob.from_dict(
        data={
            "id_int": [1.0, 2.0, 4.0, 5.0],
            "id_str": ["a", "b", "c", None],
            "x": [1, 2, 3, 4],
        },
        request=retrieval_request,
    )
    facts = RetrievalJob.from_dict(
        data={"id_int": [1.0, 2.0, 4.0, 5.0], "id_str": ["a", "b", "c", None]},
        request=retrieval_request,
    )
    config = RedisConfig.localhost()
    source = RedisSource(config)

    await source.insert(insert_facts, retrieval_request)

    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrieval_request],
        facts=facts,
        formatter=DateFormatter.iso_8601(),
    )
    data = await job.to_lazy_polars()

    assert (
        data.collect().select("x").to_series().equals(pl.Series("x", [1, 2, 3, None]))
    )
