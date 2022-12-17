import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Coroutine

import click
from pytz import utc  # type: ignore

from aligned.compiler.repo_reader import RepoReader
from aligned.feature_source import WritableFeatureSource
from aligned.schemas.codable import Codable
from aligned.schemas.feature import Feature
from aligned.schemas.repo_definition import RepoDefinition


def sync(method: Coroutine) -> Any:
    return asyncio.get_event_loop().run_until_complete(method)


def make_tzaware(t: datetime) -> datetime:
    """We assume tz-naive datetimes are UTC"""
    if t.tzinfo is None:
        return t.replace(tzinfo=utc)
    else:
        return t


def load_envs(path: Path) -> None:
    if path.is_file():
        import os

        with path.open() as file:
            for line in file:
                if len(line) < 3:
                    continue
                key, value = line.strip().split('=')
                os.environ[key] = value
    else:
        click.echo(f'No env file found at {path}')


@click.group()
def cli() -> None:
    pass


@cli.command('apply')
@click.option(
    '--repo-path',
    default='.',
    help='The path to the repo',
)
@click.option(
    '--reference-file',
    default='feature_store_location.py',
    help='The path to a feature store reference file. Defining where to read and write the feature store',
)
@click.option(
    '--env-file',
    default='.env',
    help='The path to env variables',
)
def apply_command(repo_path: str, reference_file: str, env_file: str) -> None:
    """
    Create or update a feature store deployment
    """

    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    reference_file_path = Path(reference_file).absolute()
    load_envs(dir / env_file)
    sys.path.append(str(dir))
    repo_ref = RepoReader.reference_from_path(dir, reference_file_path)

    click.echo(f'Updating file at: {repo_ref.selected}')

    # old_def = sync(repo_ref.selected_file.feature_store())

    repo_def = sync(RepoReader.definition_from_path(dir))

    if file := repo_ref.selected_file:
        sync(file.write(repo_def.to_json(omit_none=True).encode('utf-8')))
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
    default='127.0.0.1',
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
    default='feature_store_location:feature_server',
    help='The path to the feature store server',
)
def serve_command(
    repo_path: str, host: str, port: int, workers: int, env_file: str, reload: bool, server_path: str
) -> None:
    """
    Starts a API serving the feature store
    """
    from logging.config import dictConfig

    import uvicorn

    handler = 'console'
    log_format = '%(levelname)s:\t\b%(asctime)s %(name)s:%(lineno)d [%(correlation_id)s] %(message)s'
    dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'filters': {
                'correlation_id': {
                    '()': 'asgi_correlation_id.CorrelationIdFilter',
                    'uuid_length': 16,
                },
            },
            'formatters': {
                'console': {'class': 'logging.Formatter', 'datefmt': '%H:%M:%S', 'format': log_format}
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'filters': ['correlation_id'],
                    'formatter': 'console',
                }
            },
            'loggers': {
                # project
                '': {'handlers': [handler], 'level': 'INFO', 'propagate': True},
            },
        }
    )
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
@click.option(
    '--repo-path',
    default='.',
    help='The path to the repo',
)
@click.option(
    '--reference-file',
    default='feature_store_location.py',
    help='The path to a feature store reference file. Defining where to read and write the feature store',
)
@click.option(
    '--workers',
    '-w',
    default=1,
    help='The number of workers',
)
@click.option('--views', '-v', help='The views to run in a worker', multiple=True)
@click.option(
    '--env-file',
    default='.env',
    help='The path to env variables',
)
def serve_worker_command(
    repo_path: str, reference_file: str, views: list[str], workers: int, env_file: str
) -> None:
    """
    Starts a API serving the feature store
    """
    from logging.config import dictConfig

    from aligned.worker import start

    handler = 'console'
    log_format = '%(levelname)s:\t\b%(asctime)s %(name)s:%(lineno)d [%(correlation_id)s] %(message)s'
    dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'filters': {
                'correlation_id': {
                    '()': 'asgi_correlation_id.CorrelationIdFilter',
                    'uuid_length': 16,
                },
            },
            'formatters': {
                'console': {'class': 'logging.Formatter', 'datefmt': '%H:%M:%S', 'format': log_format}
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'filters': ['correlation_id'],
                    'formatter': 'console',
                }
            },
            'loggers': {
                # project
                '': {'handlers': [handler], 'level': 'INFO', 'propagate': True},
            },
        }
    )
    # Needed in order to find the feature_store_location file
    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    reference_file_path = Path(reference_file).absolute()
    sys.path.append(str(dir))
    env_file_path = dir / env_file
    load_envs(env_file_path)

    repo_ref = RepoReader.reference_from_path(dir, reference_file_path)

    if not repo_ref.selected_file:
        raise ValueError('No selected feature store in the repo reference. Make sure the env var is set')

    feature_store = sync(repo_ref.selected_file.feature_store())

    sync(start(store=feature_store, feature_views_to_process=set(views)))


@cli.command('materialize')
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
@click.option(
    '--days',
    help='The number of days to materialize',
)
@click.option(
    '--view',
    help='The feature view to materialize',
)
def materialize_command(repo_path: str, env_file: str, days: str, view: str) -> None:
    """
    Materializes the feature store
    """
    from aligned.feature_store import FeatureStore

    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    load_envs(dir / env_file)

    sys.path.append(str(dir))
    repo_def = sync(RepoDefinition.from_path(repo_path))
    store = FeatureStore.from_definition(repo_def)
    batch_store = store.offline_store()

    if not isinstance(store.feature_source, WritableFeatureSource):
        raise ValueError('Batch feature sources are not supported for materialization')

    number_of_days = int(days)
    views = [view] if view else list(store.feature_views.keys())

    click.echo(f'Materializing the last {number_of_days} days')
    for feature_view in views:
        fv_store = batch_store.feature_view(feature_view)
        click.echo(f'Materializing {feature_view}')
        sync(
            store.feature_source.write(
                fv_store.previous(days=number_of_days), fv_store.view.request_all.needed_requests
            )
        )


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
def profile(repo_path: str, reference_file: str, env_file: str, output: str, dataset_size: int) -> None:
    import numpy as np
    from pandas import DataFrame

    from aligned import FeatureStore

    # Make sure modules can be read, and that the env is set
    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    sys.path.append(str(dir))
    env_file_path = dir / env_file
    load_envs(env_file_path)

    online_store: FeatureStore = sync(FeatureStore.from_reference_at_path(repo_path, reference_file))
    feature_store = online_store.offline_store()

    results = ProfilingResult(numeric_features={}, categorical_features={})

    for feature_view_name in sorted(feature_store.feature_views.keys()):
        click.echo(f'Profiling: {feature_view_name}')
        feature_view = feature_store.feature_view(feature_view_name)
        data_set: DataFrame = sync(feature_view.all(limit=dataset_size).to_df())

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{asctime} {message}', style='{')
    cli()
