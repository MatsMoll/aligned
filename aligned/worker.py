from __future__ import annotations

import asyncio
import logging
import timeit
from dataclasses import dataclass, field
from pathlib import Path

from aligned.active_learning.selection import ActiveLearningMetric, ActiveLearningSelection
from aligned.active_learning.write_policy import ActiveLearningWritePolicy
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.feature_store import FeatureViewStore, ModelFeatureStore
from aligned.local.source import StorageFileReference
from aligned.online_source import OnlineSource
from aligned.redis.config import RedisStreamSource
from aligned.redis.stream import RedisStream
from aligned.retrival_job import RetrivalJob, StreamAggregationJob

logger = logging.getLogger(__name__)

# Very experimental, so can contain a lot of bugs


@dataclass
class ActiveLearningConfig:
    metric: ActiveLearningMetric
    selection: ActiveLearningSelection
    write_policy: ActiveLearningWritePolicy
    model_names: list[str] | None = None


@dataclass
class StreamWorker:

    feature_store_reference: StorageFileReference
    online_source: OnlineSource
    views_to_process: set[str]
    should_prune_unused_features: bool = field(default=False)
    active_learning_configs: list[ActiveLearningConfig] = field(default_factory=list)

    @staticmethod
    def from_reference(
        source: StorageFileReference, online_source: OnlineSource, views_to_process: set[str] | None = None
    ) -> StreamWorker:
        """
        Creates a stream worker.

        This object can start a background worker process of streaming data.

        Args:
            source (StorageFileReference): The storage of the feature store file
            online_source (OnlineSource): Where to store the processed features
            views_to_process (set[str] | None, optional): The views to process.
                Defaults to None aka, all streaming views.

        Returns:
            StreamWorker | None: A worker that can start processing
        """

        return StreamWorker(source, online_source, views_to_process)

    @staticmethod
    def from_object(repo: Path, file: Path, obj: str) -> StreamWorker:
        from aligned.compiler.repo_reader import import_module, path_to_py_module

        module_path = path_to_py_module(file, repo)
        module = import_module(module_path)

        try:
            worker = getattr(module, obj)
            if isinstance(worker, StreamWorker):
                return worker
            raise ValueError('No reference found')
        except AttributeError:
            raise ValueError('No reference found')

    def generate_active_learning_dataset(
        self,
        metric: ActiveLearningMetric | None = None,
        selection: ActiveLearningSelection | None = None,
        write_policy: ActiveLearningWritePolicy | None = None,
        model_names: list[str] | None = None,
    ) -> StreamWorker:
        self.active_learning_configs.append(
            ActiveLearningConfig(
                metric=metric or ActiveLearningMetric.max_probability(),
                selection=selection or ActiveLearningSelection.under_threshold(0.5),
                write_policy=write_policy or ActiveLearningWritePolicy.sample_size(10, 1000),
                model_names=model_names,
            )
        )
        return self

    async def start(self, should_prune_unused_features: bool) -> None:
        from aligned.data_source.stream_data_source import HttpStreamSource

        store = await self.feature_store_reference.feature_store()
        store = store.with_source(self.online_source)

        views = self.views_to_process or set()
        if not self.views_to_process:
            views = [
                view.name
                for view in store.feature_views.values()
                if view.stream_data_source is not None
                and not isinstance(view.stream_data_source, HttpStreamSource)
            ]
        if not views:
            raise ValueError('No feature views with streaming source to process')

        feature_views_to_process = views

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
            process_views = views
            if should_prune_unused_features:
                process_views = [view.with_optimised_write() for view in process_views]
            processes.append(process(redis_stream, topic_name, process_views))

        for active_learning_config in self.active_learning_configs:
            for model_name in set(active_learning_config.model_names):
                processes.append(
                    process_predictions(redis_stream, store.model(model_name), active_learning_config)
                )

        await asyncio.gather(*processes)


async def process_predictions(
    stream_source: RedisStream, model: ModelFeatureStore, active_learning_config: ActiveLearningConfig | None
) -> None:
    from aligned.active_learning.job import ActiveLearningJob

    if not active_learning_config:
        logger.info('No active learning config found, will not listen to predictions')
        return

    topic_name = model.model.predictions_view.stream_source.topic_name
    last_id = '$'
    logger.info(f'Started listning to {topic_name}')

    while True:
        stream_values = await stream_source.read_from_timestamp({topic_name: last_id})

        if not stream_values:
            continue
        start_time = timeit.default_timer()
        _, values = stream_values[0]
        last_id = values[-1][0]
        records = [value for _, value in values]

        request = model.model.request_all_predictions.needed_requests[0]
        job = RetrivalJob.from_dict(records, request).ensure_types([request])
        job = ActiveLearningJob(
            job,
            model.model,
            active_learning_config.metric,
            active_learning_config.selection,
            active_learning_config.write_policy,
        )
        _ = await job.to_polars()

        logger.info(f'Processing {len(records)} predictions in {timeit.default_timer() - start_time} seconds')


async def single_processing(
    stream_source: RedisStream, topic_name: str, feature_view: FeatureViewStore
) -> None:
    last_id = '$'
    logger.info(f'Started listning to {topic_name}')
    # request = feature_view.view.request_all.needed_requests[0]
    # aggregations = request.aggregate_over()
    # checkpoints = {
    #     window: FileSource.parquet_at(f'{feature_view.view.name}_agg_{window.time_window.total_seconds()}')
    #     for window in aggregations.keys()
    # }
    while True:
        stream_values = await stream_source.read_from_timestamp({topic_name: last_id})

        if not stream_values:
            continue
        start_time = timeit.default_timer()
        _, values = stream_values[0]
        last_id = values[-1][0]
        records = [record for _, record in values]

        job = stream_job(records, feature_view)

        await feature_view.batch_write(job)  # type: ignore [arg-type]
        elapsed = timeit.default_timer() - start_time
        logger.info(f'Wrote {len(values)} records in {elapsed} seconds')


def stream_job(values: list[dict], feature_view: FeatureViewStore) -> RetrivalJob:
    from aligned import FileSource

    request = feature_view.request
    job = (
        RetrivalJob.from_dict(values, request)
        .validate_entites()
        .fill_missing_columns()
        .ensure_types([request])
    )

    aggregations = request.aggregate_over()
    if not aggregations:
        return job

    checkpoints = {}

    for aggregation in aggregations.keys():
        name = f'{feature_view.view.name}_agg'
        if aggregation.window:
            time_window = aggregation.window
            name += f'_{time_window.time_window.total_seconds()}'
        checkpoints[aggregation] = FileSource.parquet_at(name)

    return StreamAggregationJob(job, checkpoints)


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

        await asyncio.gather(*[view.batch_write(stream_job(mapped_values, view)) for view in feature_views])


async def process(stream_source: RedisStream, topic_name: str, feature_views: list[FeatureViewStore]) -> None:
    if len(feature_views) == 1:
        await single_processing(stream_source, topic_name, feature_views[0])
    else:
        await multi_processing(stream_source, topic_name, feature_views)
