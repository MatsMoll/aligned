from pathlib import Path
import click
from aladdin.repo_definition import RepoDefinition
import sys
from datetime import datetime
from pytz import utc

def make_tzaware(t: datetime) -> datetime:
    """We assume tz-naive datetimes are UTC"""
    if t.tzinfo is None:
        return t.replace(tzinfo=utc)
    else:
        return t

@click.group()
def cli():
    pass


@cli.command("apply")
@click.option(
    "--repo-path",
    default=".",
    help="The path to the repo",
)
@click.option(
    "--repo-file",
    default="repo_definition.json",
    help="The repo definition file path",
)
def apply_command(repo_path: str, repo_file: str):
    """
    Create or update a feature store deployment
    """
    from aladdin.repo_reader import RepoReader
    
    import os

    dir = Path.cwd() if repo_path == "." else Path(repo_path).absolute()
    repo_def = RepoReader.from_path(dir)
    
    with open(repo_file, "w") as file:
        # This will not work with S3 implementations, so an abstraction layer is needed
        file.write(repo_def.to_json())
    

@cli.command("plan")
@click.option(
    "--repo-path",
    default=".",
    help="The path to the repo",
)
def plan_command(repo_path: str):
    """
    Prints the plan for updating the feature store file
    """
    from aladdin.repo_reader import RepoReader

    dir = Path.cwd() if repo_path == "." else Path(repo_path).absolute()
    click.echo(RepoReader.from_path(dir))

@cli.command("serve")
@click.option(
    "--repo-path",
    default=".",
    help="The path to the repo",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="The host to serve on",
)
@click.option(
    "--port",
    "-p",
    default=8000,
    help="The port to serve on",
)
@click.option(
    "--workers",
    "-w",
    default=1,
    help="The number of workers",
)
@click.option(
    "--repo-file",
    default="feature_store.yaml",
    help="The path to the feature store file",
)
def serve_command(repo_path: str, host: str, port: int, repo_file: str, workers: int):
    """
    Starts a API serving the feature store
    """
    from aladdin.feature_store import FeatureStore
    from aladdin.server import FastAPIServer

    dir = Path.cwd() if repo_path == "." else Path(repo_path).absolute()
    sys.path.append(str(dir))
    repo_def = RepoDefinition.from_url(repo_file)
    store = FeatureStore.from_definition(repo_def)
    FastAPIServer.run(store, host, port, workers)


if __name__ == "__main__":
    cli()