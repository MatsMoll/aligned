from __future__ import annotations

import asyncio
import logging
import timeit
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

from prometheus_client import Counter, Histogram

from aligned.active_learning.selection import ActiveLearningMetric, ActiveLearningSelection
from aligned.active_learning.write_policy import ActiveLearningWritePolicy
from aligned.data_source.batch_data_source import ColumnFeatureMappable
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.feature_source import WritableFeatureSource
from aligned.feature_store import FeatureStore, FeatureViewStore, ModelFeatureStore
from aligned.retrival_job import RetrivalJob, StreamAggregationJob
from aligned.sources.local import AsRepoDefinition
from aligned.streams.interface import ReadableStream

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

processed_rows_count = Counter(
    name='aligned_processed_rows', documentation='Number of processed rows', labelnames=['feature_view']
)
process_time = Histogram(
    'aligned_process_time', documentation='Time used to process the records', labelnames=['feature_view']
)

# Very experimental, so can contain a lot of bugs


@dataclass
class ActiveLearningConfig:
    metric: ActiveLearningMetric
    selection: ActiveLearningSelection
    write_policy: ActiveLearningWritePolicy
    model_names: list[str] | None = None


@dataclass
class StreamWorker:

    repo_definition: AsRepoDefinition
    sink_source: WritableFeatureSource
    views_to_process: set[str] | None = field(default=None)
    should_prune_unused_features: bool = field(default=False)
    active_learning_configs: list[ActiveLearningConfig] = field(default_factory=list)
    metric_logging_port: int | None = field(default=None)

    read_timestamps: dict[str, str] = field(default_factory=dict)
    default_start_timestamp: str | None = field(default=None)

    @staticmethod
    def from_reference(
        source: AsRepoDefinition,
        sink_source: WritableFeatureSource,
        views_to_process: set[str] | None = None,
    ) -> StreamWorker:
        """
        Creates a stream worker.

        This object can start a background worker process of streaming data.

        Args:
            source (StorageFileReference): The storage of the feature store file
            sink_source (WritableFeatureSource): Where to store the processed features
            views_to_process (set[str] | None, optional): The views to process.
                Defaults to None aka, all streaming views.

        Returns:
            StreamWorker | None: A worker that can start processing
        """

        return StreamWorker(source, sink_source, views_to_process)

    @staticmethod
    def from_object(repo: Path, file: Path, obj: str) -> StreamWorker:
        from aligned.compiler.repo_reader import import_module, path_to_py_module
        from aligned.exceptions import StreamWorkerNotFound

        module_path = path_to_py_module(file, repo)

        try:
            module = import_module(module_path)

            worker = getattr(module, obj)
            if isinstance(worker, StreamWorker):
                return worker
            raise ValueError('No reference found')
        except AttributeError:
            raise ValueError('No reference found')
        except ModuleNotFoundError:
            raise StreamWorkerNotFound(module_path)

    def generate_active_learning_dataset(
        self,
        metric: ActiveLearningMetric | None = None,
        selection: ActiveLearningSelection | None = None,
        write_policy: ActiveLearningWritePolicy | None = None,
        model_names: list[str] | None = None,
    ) -> StreamWorker:
        self.active_learning_configs.append(
            ActiveLearningConfig(
                metric=metric or ActiveLearningMetric.max_confidence(),
                selection=selection or ActiveLearningSelection.under_threshold(0.5),
                write_policy=write_policy or ActiveLearningWritePolicy.sample_size(10, 1000),
                model_names=model_names,
            )
        )
        return self

    def read_from_timestamps(self, timestamps: dict[str, str]) -> StreamWorker:
        self.read_timestamps = timestamps
        return self

    def read_from(self, timestamp: str) -> StreamWorker:
        self.default_start_timestamp = timestamp
        return self

    def expose_metrics_at(self, port: int) -> StreamWorker:
        self.metric_logging_port = port
        return self

    def prune_unused_features(self, should_prune_unused_features: bool | None = None) -> StreamWorker:
        self.should_prune_unused_features = True
        if should_prune_unused_features:
            self.should_prune_unused_features = should_prune_unused_features
        return self

    def feature_views_by_topic(self, store: FeatureStore) -> dict[str, list[FeatureViewStore]]:
        from aligned.data_source.stream_data_source import HttpStreamSource

        feature_views_to_process = self.views_to_process or set()
        if not self.views_to_process:
            feature_views_to_process = {
                view.name
                for view in store.feature_views.values()
                if view.stream_data_source is not None
                and not isinstance(view.stream_data_source, HttpStreamSource)
            }
        if not feature_views_to_process:
            raise ValueError('No feature views with streaming source to process')

        feature_views: dict[str, list[FeatureViewStore]] = {}

        for view in store.feature_views.values():
            if view.name not in feature_views_to_process:
                continue
            if not view.stream_data_source:
                logger.debug(f'View: {view.name} have no stream source. Therefore, it will not be processed')
                continue

            source = view.stream_data_source

            view_store = store.feature_view(view.name)
            if self.should_prune_unused_features:
                logger.debug(f'Optimising the write for {view.name} based on model usage')
                view_store = view_store.with_optimised_write()

            request = view_store.request
            if len(request.all_features) == 0:
                logger.debug(
                    f'View: {view.name} had no features to process. Therefore, it will not be ignored'
                )
                continue

            if source.topic_name in feature_views:
                feature_views[source.topic_name] = feature_views[source.topic_name] + [view_store]
            else:
                feature_views[source.topic_name] = [view_store]

        return feature_views

    async def start(self) -> None:
        from prometheus_client import start_http_server

        store = await self.repo_definition.feature_store()
        store = store.with_source(self.sink_source)

        feature_views = self.feature_views_by_topic(store)

        processes = []
        for topic_name, views in feature_views.items():
            process_views = views
            stream: StreamDataSource = views[0].view.stream_data_source
            stream_consumer = stream.consumer(
                self.read_timestamps.get(topic_name, self.default_start_timestamp)
            )
            processes.append(process(stream_consumer, topic_name, process_views))

        for active_learning_config in self.active_learning_configs:

            if not active_learning_config.model_names:
                continue

            for model_name in set(active_learning_config.model_names):
                model = store.models[model_name]
                source = model.predictions_view.stream_source

                if not source:
                    logger.debug(f'Skipping to setup active learning set for {model_name}')

                processes.append(
                    process_predictions(source.consumer(), store.model(model_name), active_learning_config)
                )

        if self.metric_logging_port:
            start_http_server(self.metric_logging_port)

        if len(processes) == 0:
            raise ValueError('No processes to start')

        await asyncio.gather(*processes)


async def process_predictions(
    stream_source: ReadableStream,
    model: ModelFeatureStore,
    active_learning_config: ActiveLearningConfig | None,
) -> None:
    from aligned.active_learning.job import ActiveLearningJob

    if not active_learning_config:
        logger.debug('No active learning config found, will not listen to predictions')
        return

    topic_name = model.model.predictions_view.stream_source.topic_name
    logger.debug(f'Started listning to {topic_name}')

    while True:
        records = await stream_source.read()

        if not records:
            continue
        start_time = timeit.default_timer()

        request = model.model.request_all_predictions.needed_requests[0]
        job = RetrivalJob.from_dict(records, request).ensure_types([request])
        job = ActiveLearningJob(
            job,
            model.model,
            active_learning_config.metric,
            active_learning_config.selection,
            active_learning_config.write_policy,
        )
        _ = await job.to_lazy_polars()

        logger.debug(
            f'Processing {len(records)} predictions in {timeit.default_timer() - start_time} seconds'
        )


def stream_job(values: list[dict], feature_view: FeatureViewStore) -> RetrivalJob:
    from aligned import FileSource

    request = feature_view.request
    mappings: dict[str, str] | None = None

    if isinstance(feature_view.view.stream_data_source, ColumnFeatureMappable):
        mappings = feature_view.view.stream_data_source.mapping_keys

    value_job = RetrivalJob.from_dict(values, request)

    if mappings:
        value_job = value_job.rename(mappings)

    job = value_job.validate_entites().fill_missing_columns().ensure_types([request])

    aggregations = request.aggregate_over()
    if not aggregations:
        return job.derive_features()

    checkpoints = {}

    for aggregation in aggregations.keys():
        name = f'{feature_view.view.name}_agg'
        if aggregation.window:
            time_window = aggregation.window
            name += f'_{time_window.time_window.total_seconds()}'
        checkpoints[aggregation] = FileSource.parquet_at(name)

    job = StreamAggregationJob(job, checkpoints).derive_features()
    if feature_view.feature_filter:
        job = job.select_columns(feature_view.feature_filter)
    return job


async def monitor_process(values: list[dict], view: FeatureViewStore):
    job = stream_job(values, view).monitor_time_used(process_time, labels=[view.view.name])
    await view.batch_write(job)
    record_count = len(values)
    processed_rows_count.labels(view.view.name).inc(record_count)


async def multi_processing(
    stream_source: ReadableStream, topic_name: str, feature_views: list[FeatureViewStore]
) -> None:
    logger.debug(f'Started listning to {topic_name}')
    while True:
        logger.debug(f'Reading {topic_name}')
        stream_values = await stream_source.read()
        logger.debug(f'Read {topic_name} values: {len(stream_values)}')

        if not stream_values:
            continue

        await asyncio.gather(*[monitor_process(stream_values, view) for view in feature_views])


async def single_processing(
    stream_source: ReadableStream, topic_name: str, feature_view: FeatureViewStore
) -> None:
    logger.debug(f'Started listning to {topic_name}')
    while True:
        logger.debug(f'Reading {topic_name}')
        records = await stream_source.read()
        logger.debug(f'Read {topic_name} values: {len(records)}')

        if not records:
            continue
        await monitor_process(records, feature_view)


async def process(
    stream_source: ReadableStream,
    topic_name: str,
    feature_views: list[FeatureViewStore],
    error_count: int = 0,
) -> None:
    # try:
    if len(feature_views) == 1:
        await single_processing(stream_source, topic_name, feature_views[0])
    else:
        await multi_processing(stream_source, topic_name, feature_views)
    # except Exception as e:
    #     logger.error(f'Error processing {topic_name}: {type(e)} - {e}')
    #     if error_count > 5:
    #         raise e
    #     await asyncio.sleep(5)
    #     await process(stream_source, topic_name, feature_views, error_count=error_count + 1)
