import asyncio
import logging

from aligned import FeatureStore
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.feature_store import FeatureViewStore
from aligned.redis.config import RedisStreamSource
from aligned.redis.stream import RedisStream

logger = logging.getLogger(__name__)

# Very experimental, so can contain a lot of bugs


async def single_processing(
    stream_source: RedisStream, topic_name: str, feature_view: FeatureViewStore
) -> None:
    last_id = '$'
    logger.info(f'Started listning to {topic_name}')
    while True:
        stream_values = await stream_source.read_from_timestamp({topic_name: last_id})

        if not stream_values:
            continue

        _, values = stream_values[0]
        last_id = values[-1][0]
        await feature_view.batch_write([record for _, record in values])  # type: ignore [arg-type]


async def multi_processing(
    stream_source: RedisStream, topic_name: str, feature_views: list[FeatureViewStore]
) -> None:
    last_id = '$'
    logger.info(f'Started listning to {topic_name}')
    while True:
        stream_values = await stream_source.read_from_timestamp({topic_name: last_id})

        if not stream_values:
            continue

        _, values = stream_values[0]
        last_id = values[-1][0]
        mapped_values = [record for _, record in values]
        await asyncio.gather(
            *[view.batch_write(mapped_values) for view in feature_views]  # type: ignore [arg-type]
        )


async def process(stream_source: RedisStream, topic_name: str, feature_views: list[FeatureViewStore]) -> None:
    if len(feature_views) == 1:
        await single_processing(stream_source, topic_name, feature_views[0])
    else:
        await multi_processing(stream_source, topic_name, feature_views)


async def start(store: FeatureStore, feature_views_to_process: set[str]) -> None:

    if not feature_views_to_process:
        raise ValueError('No feature views set. remember to set the -v flag with the views to process')

    streams: list[StreamDataSource] = []
    feature_views: dict[str, list[FeatureViewStore]] = {}

    for view in store.feature_views.values():
        if view.name not in feature_views_to_process:
            continue

        if not view.stream_data_source:
            continue
        source = view.stream_data_source

        if not streams:
            streams.append(source)
            feature_views[source.topic_name] = [store.feature_view(view.name)]
        elif isinstance(source, type(streams[0])):
            streams.append(source)
            if source.topic_name in feature_views:
                feature_views[source.topic_name] = feature_views[source.topic_name] + [
                    store.feature_view(view.name)
                ]
            else:
                feature_views[source.topic_name] = [store.feature_view(view.name)]
        else:
            raise ValueError(
                'The grouped stream sources is not of the same type.'
                f'{view.name} was as {type(source)} expected {streams[0]}'
            )

    if not isinstance(streams[0], RedisStreamSource):
        raise ValueError(
            'Only supporting Redis Streams for worker nodes as of now. '
            f'Please contribute to the repo in order to support {type(streams[0])} on a worker node'
        )

    redis_streams: list[RedisStreamSource] = streams
    for stream in redis_streams:
        if not stream.config == redis_streams[0].config:
            raise ValueError(f'Not all stream configs for {feature_views_to_process}, is equal.')

    redis_stream = RedisStream(redis_streams[0].config.redis())
    processes = []
    for topic_name, views in feature_views.items():
        processes.append(process(redis_stream, topic_name, views))
    await asyncio.gather(*processes)
