import asyncio
import logging
import os
import sys
from contextlib import suppress
from functools import wraps
from pathlib import Path
from typing import Any

import click
import json
from pytz import utc  # type: ignore

from aligned.compiler.repo_reader import RepoReader, RepoReference
from aligned.feature_store import ContractStore
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


async def store_from_reference(reference_file: str, dir: Path | None = None) -> ContractStore:
    from aligned import FileSource

    if dir is None:
        dir = Path.cwd()

    if ':' in reference_file:
        path, obj = reference_file.split(':')
        reference_file_path = Path(path).absolute()
        repo_ref = RepoReference.reference_object(dir, reference_file_path, obj)
    else:
        repo_ref = RepoReference('const', {'const': FileSource.json_at(reference_file)})

    if file := repo_ref.selected_file:
        return await file.as_contract_store()
    else:
        raise ValueError(f'No repo file found at {dir}')


@click.group()
def cli() -> None:
    pass


@cli.command('compile')
@coro
@click.option(
    '--repo-path',
    default='.',
    help='The path to the repo',
)
@click.option(
    '--reference-file',
    default='contract_store.json',
    help='The path to a contract store reference file. Defining where to read and write the feature store',
)
@click.option(
    '--env-file',
    default='.env',
    help='The path to env variables',
)
@click.option('--ignore-file', default='.alignedignore', help='The files Aligned should ignore')
async def compile(repo_path: str, reference_file: str, env_file: str, ignore_file: str) -> None:
    """
    Create or update a feature store deployment
    """
    from aligned import FileSource

    setup_logger()

    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    ignore_path = dir / ignore_file

    load_envs(dir / env_file)
    sys.path.append(str(dir))

    excludes = []

    if ignore_path.is_file():
        excludes = ignore_path.read_text().split('\n')

    if ':' in reference_file:
        path, obj = reference_file.split(':')
        reference_file_path = Path(path).absolute()
        repo_ref = RepoReference.reference_object(dir, reference_file_path, obj)
    else:
        repo_ref = RepoReference('const', {'const': FileSource.json_at(reference_file)})

    if file := repo_ref.selected_file:
        click.echo(f'Updating file at: {file}')

        repo_def = await RepoReader.definition_from_path(dir, excludes)

        data = repo_def.to_json(omit_none=True)
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        await file.write(data_bytes)
    else:
        click.echo(f'No repo file found at {dir}')


@cli.command('check-updates')
@coro
@click.option('--updated-contract')
@click.option('--reference-contract')
@click.option('--output-format', default='markdown', help='The output format')
@click.option('--output-file', default=None, help='The output format')
async def check_updates(
    updated_contract: str,
    reference_contract: str,
    output_format: str,
    output_file: str | None,
):
    """
    Check if the current changes conflicts with an existing contract store.

    This will check if:

    1. Exposed models have the needed features.
    2. If any transformations that a model depend on have changed.
    """
    from aligned.checks import (
        check_exposed_models_have_needed_features,
        impacted_models_from_transformation_diffs,
        check_exposed_models_for_potential_distribution_shift,
        ContractStoreUpdateCheckReport,
    )

    as_markdown = output_format == 'markdown' or output_format == 'md'

    new_contract_store = await store_from_reference(updated_contract)
    old_contract_store = await store_from_reference(reference_contract)

    checks = await check_exposed_models_have_needed_features(new_contract_store)
    potential_drifts = await check_exposed_models_for_potential_distribution_shift(
        old_contract_store, new_contract_store
    )
    transformation_changes = impacted_models_from_transformation_diffs(
        new_store=new_contract_store, old_store=old_contract_store
    )
    not_ok_checks = [check for check in checks if not check.is_ok]

    report = ContractStoreUpdateCheckReport(
        needed_model_input=not_ok_checks,
        potential_distribution_shifts=potential_drifts,
        model_transformation_changes=transformation_changes,
    )

    if as_markdown:
        output = report.as_markdown()
    else:
        output = json.dumps(report)

    if output_file:
        path = Path(output_file)
        path.write_text(output)
    else:
        click.echo(output)


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
    Starts an API serving data based on a contract store.
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
    Starts a worker that process the contract store streams and store them in an online source.
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
    """
    Creates a set of vector indexes for the contract store.
    """
    from aligned import ContractStore, FileSource

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

    feature_store = ContractStore.from_definition(repo_def)

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
