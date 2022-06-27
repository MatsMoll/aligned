from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from aladdin.codable import Codable
from aladdin.feature_view.combined_view import CompiledCombinedFeatureView
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.online_source import OnlineSource

if TYPE_CHECKING:
    from aladdin.local.source import FileReference


logger = logging.getLogger(__name__)


@dataclass
class RepoReference:
    env_var_name: str
    repo_paths: dict[str, FileReference]

    @property
    def selected(self) -> str:
        import os

        return os.environ[self.env_var_name]

    @property
    def selected_file(self) -> FileReference | None:
        return self.repo_paths.get(self.selected)


@dataclass
class RepoDefinition(Codable):
    feature_views: set[CompiledFeatureView]
    combined_feature_views: set[CompiledCombinedFeatureView]
    models: dict[str, set[str]]
    online_source: OnlineSource

    @staticmethod
    async def from_file(file: FileReference) -> RepoDefinition:
        repo = await file.read()
        return RepoDefinition.from_json(repo)

    @staticmethod
    async def from_reference_at_path(path: str) -> RepoDefinition:
        from aladdin.repo_reader import RepoReader

        dir_path = Path.cwd() if path == '.' else Path(path).absolute()

        try:
            reference = RepoReader.reference_from_path(dir_path)
            if file := reference.selected_file:
                logger.info(f"Loading repo from configuration '{reference.selected}'")
                return await RepoDefinition.from_file(file)
            else:
                logger.info('Found no configuration')
        except ValueError as error:
            logger.error(f'Error when loadin repo: {error}')

        logger.info('Generating repo definition')
        return RepoReader.definition_from_path(dir_path)

    @staticmethod
    def from_path(path: str) -> RepoDefinition:
        from aladdin.repo_reader import RepoReader

        dir_path = Path.cwd() if path == '.' else Path(path).absolute()
        return RepoReader.definition_from_path(dir_path)
