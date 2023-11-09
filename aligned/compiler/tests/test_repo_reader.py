from aligned.compiler.repo_reader import RepoReader
from pathlib import Path
import pytest


@pytest.mark.asyncio
async def test_repo_reader() -> None:

    path = Path('aligned/compiler/tests')
    definitions = await RepoReader.definition_from_path(path)

    assert len(definitions.feature_views) == 1

    view = list(definitions.feature_views)[0]

    assert view.name == 'test'
    assert view.source.type_name == 'psql'
    assert len(view.derived_features) == 1
    assert len(view.features) == 2
    assert len(view.entities) == 1
