import asyncio
import sys
from datetime import datetime
from pathlib import Path

import click
from pytz import utc

from aladdin.feature_source import WritableFeatureSource
from aladdin.online_source import BatchOnlineSource
from aladdin.repo_definition import RepoDefinition
from aladdin.repo_reader import RepoReader


def sync(method):
    return asyncio.get_event_loop().run_until_complete(method)


def make_tzaware(t: datetime) -> datetime:
    """We assume tz-naive datetimes are UTC"""
    if t.tzinfo is None:
        return t.replace(tzinfo=utc)
    else:
        return t


def load_envs(path: Path):
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


async def repo_for(dir: Path) -> RepoDefinition:
    repo_def: RepoDefinition = None
    if reference := RepoReader.reference_from_path(dir):
        if file := reference.selected_file:
            click.echo(f"Loading repo from configuration '{reference.selected}'")
            repo_def = await RepoDefinition.from_file(file)
        else:
            click.echo(f"Invalid repo configuration '{reference.selected}'")

    if repo_def is None:
        click.echo('Generating repo definition')
        repo_def = RepoReader.definition_from_path(dir)
    return repo_def


@click.group()
def cli():
    pass


@cli.command('apply')
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
def apply_command(repo_path: str, env_file):
    """
    Create or update a feature store deployment
    """
    from aladdin.repo_reader import RepoReader

    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    load_envs(dir / env_file)
    sys.path.append(str(dir))
    repo_def = RepoReader.definition_from_path(dir)
    repo_ref = RepoReader.reference_from_path(dir)

    if file := repo_ref.selected_file:
        sync(file.write(repo_def.to_json().encode('utf-8')))
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
def plan_command(repo_path: str, env_file: str):
    """
    Prints the plan for updating the feature store file
    """
    from aladdin.repo_reader import RepoReader

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
def serve_command(repo_path: str, host: str, port: int, workers: int, env_file: str):
    """
    Starts a API serving the feature store
    """
    from aladdin.feature_store import FeatureStore
    from aladdin.server import FastAPIServer

    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    load_envs(dir / env_file)
    sys.path.append(str(dir))
    repo_def = sync(repo_for(dir))

    store = FeatureStore.from_definition(repo_def)
    FastAPIServer.run(store, host, port, workers)


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
def materialize_command(repo_path: str, env_file: str, days: str, view: str):
    """
    Materializes the feature store
    """
    from aladdin.feature_store import FeatureStore

    dir = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
    load_envs(dir / env_file)

    sys.path.append(str(dir))
    repo_def = sync(repo_for(dir))
    store = FeatureStore.from_definition(repo_def)
    batch_store = FeatureStore.from_definition(repo_def)
    batch_store.feature_source = BatchOnlineSource().feature_source(list(store.feature_views.values()))

    if not isinstance(store.feature_source, WritableFeatureSource):
        raise ValueError('Batch feature sources are not supported for materialization')

    days = int(days)
    views = [view] if view else list(store.feature_views.keys())

    click.echo(f'Materializing the last {days} days')
    for feature_view in views:
        fv_store = batch_store.feature_view(feature_view)
        click.echo(f'Materializing {feature_view}')
        sync(
            store.feature_source.write(
                fv_store.previous(days=days), fv_store.view.request_all.needed_requests
            )
        )


if __name__ == '__main__':
    cli()
