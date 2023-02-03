import numpy as np
import pytest
from redis.asyncio.client import Pipeline  # type: ignore

from aligned.redis.config import RedisConfig
from aligned.redis.job import FactualRedisJob
from aligned.request.retrival_request import RetrivalRequest
from aligned.schemas.feature import Feature, FeatureType


@pytest.fixture
def retrival_request() -> RetrivalRequest:
    return RetrivalRequest(
        location='test',
        entities={
            Feature(name='id_int', dtype=FeatureType('').int32),
            Feature(name='id_str', dtype=FeatureType('').string),
        },
        features={
            Feature(name='x', dtype=FeatureType('').int32),
        },
        derived_features=set(),
        event_timestamp=None,
    )


@pytest.mark.asyncio
async def test_factual_redis_job(mocker, retrival_request) -> None:  # type: ignore[no-untyped-def]
    values = ['20', '44']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrival_request],
        facts={'id_int': [1.0, 2.0, None, 4.0], 'id_str': ['a', 'b', 'c', None]},
    )

    result = await job.to_pandas()
    redis_mock.assert_called_once()
    x_result = [int(value) for value in values] + [0, 0]
    assert np.all(result['x'].fillna(0).values == x_result), f'Got {result}'


@pytest.mark.asyncio
async def test_factual_redis_job_int_as_str(mocker, retrival_request) -> None:  # type: ignore[no-untyped-def]
    values = ['20', '44']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrival_request],
        facts={'id_int': ['1', '2', None, '4'], 'id_str': ['a', 'b', 'c', None]},
    )

    result = await job.to_pandas()
    redis_mock.assert_called_once()
    x_result = [int(value) for value in values] + [0, 0]
    assert np.all(result['x'].fillna(0).values == x_result)


@pytest.mark.asyncio
async def test_nan_entities_job(mocker, retrival_request) -> None:  # type: ignore[no-untyped-def]
    values = ['20', '44']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrival_request],
        facts={'id_int': [None, '4'], 'id_str': ['c', None]},
    )

    _ = await job.to_pandas()
    redis_mock.assert_not_called()


@pytest.mark.asyncio
async def test_no_entities_job(mocker, retrival_request) -> None:  # type: ignore[no-untyped-def]
    values = ['20', '44']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrival_request],
        facts={'id_int': [], 'id_str': []},
    )

    _ = await job.to_pandas()
    redis_mock.assert_not_called()


@pytest.mark.asyncio
async def test_factual_redis_job_int_entity(mocker) -> None:  # type: ignore[no-untyped-def]

    retrival_request = RetrivalRequest(
        location='test',
        entities={Feature(name='id_int', dtype=FeatureType('').int32)},
        features={
            Feature(name='x', dtype=FeatureType('').int32),
        },
        derived_features=set(),
        event_timestamp=None,
    )

    values = ['20', '44', '55']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    job = FactualRedisJob(
        RedisConfig.localhost(),
        requests=[retrival_request],
        facts={'id_int': [1.0, 2.0, 4.0, None]},
    )

    result = await job.to_pandas()
    redis_mock.assert_called_once()
    x_result = [int(value) for value in values] + [0]
    assert np.all(result['x'].fillna(0).values == x_result)
