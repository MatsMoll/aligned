import asyncio
import logging
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import click
from pytz import utc  # type: ignore

from aligned.compiler.repo_reader import RepoReader, RepoReference
from aligned.schemas.codable import Codable
from aligned.schemas.feature import Feature
from aligned.worker import StreamWorker
from collections.abc import Callable
from datetime import datetime


def coro(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


def make_tzaware(t: datetime) -> datetime:
    """We assume tz-naive datetimes are UTC"""
    if t.tzinfo is None:
        return t.replace(tzinfo=utc)
    else:
        return t


def load_envs(path: Path) -> None:
    from dotenv import load_dotenv

    if not load_dotenv(path):
        click.echo(f'No env file found at {path}')


def setup_logger():
    from importlib.util import find_spec
    from logging.config import dictConfig

    handler = 'console'
    log_format = '%(levelname)s:\t\b%(asctime)s %(name)s:%(lineno)d %(message)s'
    configs = {
        'version': 1,
        'disable_existing_loggers': False,
        'filters': {},
        'formatters': {
            'console': {'class': 'logging.Formatter', 'datefmt': '%H:%M:%S', 'format': log_format}
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'filters': [],
                'formatter': 'console',
            }
        },
        'loggers': {
            # project
            '': {'handlers': [handler], 'level': 'INFO', 'propagate': True},
        },
    }

    if find_spec('asgi_correlation_id'):
        log_format = '%(levelname)s:\t\b%(asctime)s %(name)s:%(lineno)d [%(correlation_id)s] %(message)s'
        configs['filters']['correlation_id'] = {
            '()': 'asgi_correlation_id.CorrelationIdFilter',
            'uuid_length': 16,
        }
        configs['handlers']['console']['filters'].append('correlation_id')
        configs['formatters']['console']['format'] = log_format

    dictConfig(configs)


@click.group()
def cli() -> None:
    pass


@cli.command('apply')
@coro
@click.option(
    '--repo-path',
    default='.',
    help='The path to the repo',
)
@click.option(
    '--reference-file',
    default='source.py:source',
    help='The path to a feature store reference file. Defining where to read and write the feature store',
)
@click.option(
    '--env-file',
    default='.env',
    help='The path to env variables',
)
@click.option('--ignore-file', default='.alignedignore', help='The files Aligned should ignore')
async def apply_command(repo_path: str, reference_file: str, env_file: str, ignore_file: str) -> None:
    """
    Create or update a feature store deployment
    """
    from aligned import FileSource

    setup_logger()

    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    ignore_path = dir / ignore_file

    path, obj = reference_file.split(':')
    reference_file_path = Path(path).absolute()
    load_envs(dir / env_file)
    sys.path.append(str(dir))

    excludes = []

    if ignore_path.is_file():
        excludes = ignore_path.read_text().split('\n')

    repo_ref = RepoReference('const', {'const': FileSource.json_at('./feature-store.json')})
    with suppress(ValueError):
        repo_ref = RepoReference.reference_object(dir, reference_file_path, obj)

    if file := repo_ref.selected_file:
        click.echo(f'Updating file at: {file}')

        repo_def = await RepoReader.definition_from_path(dir, excludes)

        await file.write(repo_def.to_json(omit_none=True).encode('utf-8'))
    else:
        click.echo(f'No repo file found at {dir}')


@cli.command('plan')
@click.option(
    '--repo-path',
    default='.',
    help='The path to the repo',
)
@click.option(
    '--env-file',
    default='.env',
    help='The path to env variables',
)
def plan_command(repo_path: str, env_file: str) -> None:
    """
    Prints the plan for updating the feature store file
    """

    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    sys.path.append(str(dir))
    load_envs(dir / env_file)
    click.echo(RepoReader.definition_from_path(dir))


@cli.command('serve')
@click.option(
    '--repo-path',
    default='.',
    help='The path to the repo',
)
@click.option(
    '--host',
    default=None,
    help='The host to serve on',
)
@click.option(
    '--port',
    '-p',
    default=8000,
    help='The port to serve on',
)
@click.option(
    '--workers',
    '-w',
    default=1,
    help='The number of workers',
)
@click.option(
    '--env-file',
    default='.env',
    help='The path to env variables',
)
@click.option(
    '--reload',
    '-r',
    default=False,
    help='If the server should reload on dir changes',
)
@click.option(
    '--server-path',
    default='server:server',
    help='The path to the feature store server',
)
def serve_command(
    repo_path: str, port: int, workers: int, env_file: str, reload: bool, server_path: str, host: str | None
) -> None:
    """
    Starts a API serving the feature store
    """
    import uvicorn

    setup_logger()

    host = host or os.getenv('HOST', '127.0.0.1')

    os.environ['ALADDIN_ENABLE_SERVER'] = 'True'
    # Needed in order to find the feature_store_location file
    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    sys.path.append(str(dir))
    env_file_path = dir / env_file
    load_envs(env_file_path)
    uvicorn.run(
        server_path,
        host=host or '127.0.0.1',
        port=port or 8000,
        workers=workers or workers,
        reload=reload,
        env_file=env_file_path,
    )


@cli.command('serve-worker')
@coro
@click.option(
    '--repo-path',
    default='.',
    help='The path to the repo',
)
@click.option(
    '--worker-path',
    default='worker.py:worker',
    help='The path to the `StreamWorker`',
)
@click.option(
    '--env-file',
    default='.env',
    help='The path to env variables',
)
async def serve_worker_command(repo_path: str, worker_path: str, env_file: str) -> None:
    """
    Starts a API serving the feature store
    """
    setup_logger()

    # Needed in order to find the feature_store_location file
    path, obj = worker_path.split(':')
    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    reference_file_path = Path(path).absolute()
    sys.path.append(str(dir))
    env_file_path = dir / env_file
    load_envs(env_file_path)

    worker = StreamWorker.from_object(dir, reference_file_path, obj)

    await worker.start()


@dataclass
class CategoricalFeatureSummary(Codable):
    missing_percentage: float
    unique_values: int
    values: list[str]
    value_count: list[int]


@dataclass
class NumericFeatureSummary(Codable):
    missing_percentage: float
    mean: float | None
    median: float | None
    std: float | None
    lowest: float | None
    highests: float | None
    histogram_count: list[int]
    histogram_splits: list[float]


@dataclass
class ProfilingResult(Codable):
    numeric_features: dict[str, NumericFeatureSummary]
    categorical_features: dict[str, CategoricalFeatureSummary]


# Should add some way of profiling models, not feature views.
# Or maybe both
@cli.command('profile')
@coro
@click.option(
    '--repo-path',
    default='.',
    help='The path to the repo',
)
@click.option(
    '--reference-file',
    default='feature_store_location.py',
    help='The file defining where to read the feature store from',
)
@click.option('--output', default='profiling-result.json')
@click.option('--dataset-size', default=10000)
@click.option(
    '--env-file',
    default='.env',
    help='The path to env variables',
)
async def profile(repo_path: str, reference_file: str, env_file: str, output: str, dataset_size: int) -> None:
    import numpy as np
    from pandas import DataFrame

    from aligned import FeatureStore

    # Make sure modules can be read, and that the env is set
    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    sys.path.append(str(dir))
    env_file_path = dir / env_file
    load_envs(env_file_path)

    online_store: FeatureStore = await FeatureStore.from_reference_at_path(repo_path, reference_file)
    feature_store = online_store.offline_store()

    results = ProfilingResult(numeric_features={}, categorical_features={})

    for feature_view_name in sorted(feature_store.feature_views.keys()):
        click.echo(f'Profiling: {feature_view_name}')
        feature_view = feature_store.feature_view(feature_view_name)
        data_set: DataFrame = feature_view.all(limit=dataset_size).to_pandas()

        all_features: list[Feature] = list(feature_view.view.features) + list(
            feature_view.view.derived_features
        )
        for feature in all_features:

            data_slice = data_set[feature.name]

            reference = f'{feature_view_name}:{feature.name}'

            if (not feature.dtype.is_numeric) or feature.dtype.name == 'bool':
                unique_values = data_slice.unique()
                filter_unique_nan_values = [
                    value
                    for value in unique_values
                    if not (
                        str(value).lower() == 'nan' or str(value).lower() == 'nat' or str(value) == '<NA>'
                    )
                ]

                results.categorical_features[reference] = CategoricalFeatureSummary(
                    missing_percentage=(data_slice.isna() | data_slice.isnull()).sum() / data_slice.shape[0],
                    unique_values=unique_values.shape[0],
                    values=[str(value) for value in filter_unique_nan_values],
                    value_count=data_slice.value_counts()[filter_unique_nan_values].tolist(),
                )
            else:
                description = data_slice.describe()
                n_bins = np.min([50, len(data_slice.unique())])
                max_value = description['max']
                min_value = description['min']

                if np.isnan(max_value):
                    continue

                width = (max_value - min_value) / n_bins

                if width <= 0:
                    histogram = [description['count']]
                    cuts = []
                else:
                    cuts = np.arange(start=min_value, stop=max_value + width, step=width)
                    histogram, _ = np.histogram(data_slice.loc[~data_slice.isna()].values, cuts)

                results.numeric_features[reference] = NumericFeatureSummary(
                    missing_percentage=(data_slice.isna() | data_slice.isnull()).sum() / data_slice.shape[0],
                    mean=description['mean'] if not np.isnan(description['mean']) else None,
                    median=description['50%'] if not np.isnan(description['50%']) else None,
                    std=description['std'] if not np.isnan(description['std']) else None,
                    lowest=description['min'] if not np.isnan(description['min']) else None,
                    highests=description['max'] if not np.isnan(description['max']) else None,
                    histogram_count=list(histogram),
                    histogram_splits=list(cuts),
                )

    Path(output).write_bytes(results.to_json().encode('utf-8'))


@cli.command('create-indexes')
@coro
@click.option(
    '--repo-path',
    default='.',
    help='The path to the repo',
)
@click.option(
    '--reference-file',
    default='feature_store_location.py:source',
    help='The path to a feature store reference file. Defining where to read and write the feature store',
)
@click.option(
    '--env-file',
    default='.env',
    help='The path to env variables',
)
async def create_indexes(repo_path: str, reference_file: str, env_file: str) -> None:
    from aligned import FeatureStore, FileSource

    setup_logger()

    # Make sure modules can be read, and that the env is set
    path, obj = reference_file.split(':')
    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    reference_file_path = Path(path)

    sys.path.append(str(dir))
    env_file_path = dir / env_file
    load_envs(env_file_path)

    repo_ref = RepoReference('const', {'const': FileSource.json_at('./feature-store.json')})
    with suppress(ValueError):
        repo_ref = RepoReference.reference_object(dir, reference_file_path, obj)

    if file := repo_ref.selected_file:
        click.echo(f'Updating file at: {file}')

        repo_def = await RepoReader.definition_from_path(dir)
    else:
        click.echo(f'No repo file found at {dir}. Returning without creating indexes')
        return

    feature_store = FeatureStore.from_definition(repo_def)

    for feature_view_name in sorted(feature_store.feature_views.keys()):
        view = feature_store.feature_views[feature_view_name]
        if view.indexes is None:
            continue

        for index in view.indexes:
            click.echo(f'Creating indexes for: {feature_view_name}')
            await index.storage.create_index(index)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{asctime} {message}', style='{')
    cli()
