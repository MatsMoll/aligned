import numpy as np
import polars as pl
import pytest
from redis.asyncio.client import Pipeline  # type: ignore

from aligned.redis.job import FactualRedisJob
from aligned.request.retrival_request import RetrivalRequest
from aligned.retrival_job import RetrivalJob
from aligned.schemas.feature import Feature, FeatureLocation, FeatureType
from aligned.sources.redis import RedisConfig, RedisSource


@pytest.fixture
def retrival_request() -> RetrivalRequest:
    return RetrivalRequest(
        name='test',
        location=FeatureLocation.feature_view('test'),
        entities={
            Feature(name='id_int', dtype=FeatureType.int32()),
            Feature(name='id_str', dtype=FeatureType.string()),
        },
        features={
            Feature(name='x', dtype=FeatureType.int32()),
        },
        derived_features=set(),
        event_timestamp=None,
    )


@pytest.mark.asyncio
async def test_factual_redis_job(mocker, retrival_request) -> None:  # type: ignore[no-untyped-def]
    values = ['20', '44']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    facts = RetrivalJob.from_dict(
        data={'id_int': [1.0, 2.0, None, 4.0], 'id_str': ['a', 'b', 'c', None]},
        request=retrival_request,
    )
    job = FactualRedisJob(RedisConfig.localhost(), requests=[retrival_request], facts=facts)

    result = await job.to_pandas()
    redis_mock.assert_called_once()
    x_result = [int(value) for value in values] + [0, 0]
    assert np.all(result['x'].fillna(0).values == x_result), f'Got {result}'


@pytest.mark.asyncio
async def test_factual_redis_job_int_as_str(mocker, retrival_request) -> None:  # type: ignore[no-untyped-def]
    values = ['20', '44']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    facts = RetrivalJob.from_dict(
        data={'id_int': ['1', '2', None, '4'], 'id_str': ['a', 'b', 'c', None]},
        request=retrival_request,
    )

    job = FactualRedisJob(RedisConfig.localhost(), requests=[retrival_request], facts=facts)

    result = await job.to_pandas()
    redis_mock.assert_called_once()
    x_result = [int(value) for value in values] + [0, 0]
    assert np.all(result['x'].fillna(0).values == x_result)


@pytest.mark.asyncio
async def test_nan_entities_job(mocker, retrival_request) -> None:  # type: ignore[no-untyped-def]
    values = ['20', '44']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    facts = RetrivalJob.from_dict(
        data={'id_int': [None, '4'], 'id_str': ['c', None]},
        request=retrival_request,
    )
    job = FactualRedisJob(RedisConfig.localhost(), requests=[retrival_request], facts=facts)

    _ = await job.to_pandas()
    redis_mock.assert_not_called()


@pytest.mark.asyncio
async def test_no_entities_job(mocker, retrival_request) -> None:  # type: ignore[no-untyped-def]
    values = ['20', '44']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    facts = RetrivalJob.from_dict(
        data={'id_int': [], 'id_str': []},
        request=retrival_request,
    )
    job = FactualRedisJob(RedisConfig.localhost(), requests=[retrival_request], facts=facts)

    _ = await job.to_pandas()
    redis_mock.assert_not_called()


@pytest.mark.asyncio
async def test_factual_redis_job_int_entity(mocker) -> None:  # type: ignore[no-untyped-def]

    retrival_request = RetrivalRequest(
        name='test',
        location=FeatureLocation.feature_view('test'),
        entities={Feature(name='id_int', dtype=FeatureType.int32())},
        features={
            Feature(name='x', dtype=FeatureType.int32()),
        },
        derived_features=set(),
        event_timestamp=None,
    )

    values = ['20', '44', '55']

    redis_mock = mocker.patch.object(Pipeline, 'execute', return_value=values)

    facts = RetrivalJob.from_dict(
        data={'id_int': [1.0, 2.0, 4.0, None]},
        request=retrival_request,
    )
    job = FactualRedisJob(RedisConfig.localhost(), requests=[retrival_request], facts=facts)

    result = await job.to_pandas()
    redis_mock.assert_called_once()
    x_result = [int(value) for value in values] + [0]
    assert np.all(result['x'].fillna(0).values == x_result)


@pytest.mark.asyncio
async def test_write_job(mocker, retrival_request: RetrivalRequest) -> None:  # type: ignore[no-untyped-def]

    import fakeredis.aioredis

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)

    _ = mocker.patch.object(RedisConfig, 'redis', return_value=redis)

    insert_facts = RetrivalJob.from_dict(
        data={'id_int': [1.0, 2.0, 4.0, 5.0], 'id_str': ['a', 'b', 'c', None], 'x': [1, 2, 3, 4]},
        request=retrival_request,
    )
    facts = RetrivalJob.from_dict(
        data={'id_int': [1.0, 2.0, 4.0, 5.0], 'id_str': ['a', 'b', 'c', None]},
        request=retrival_request,
    )
    config = RedisConfig.localhost()
    source = RedisSource(config)

    await source.insert(insert_facts, [retrival_request])

    job = FactualRedisJob(RedisConfig.localhost(), requests=[retrival_request], facts=facts)
    data = await job.to_lazy_polars()

    assert data.collect().select('x').to_series().equals(pl.Series('x', [1, 2, 3, None]))
