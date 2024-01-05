from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from aligned.enricher import Enricher
from aligned.feature_source import FeatureSource, FeatureSourceFactory
from aligned.schemas.codable import Codable
from aligned.schemas.feature_view import CompiledCombinedFeatureView, CompiledFeatureView
from aligned.schemas.model import Model

if TYPE_CHECKING:
    from fastapi import FastAPI

    from aligned.sources.local import StorageFileReference


logger = logging.getLogger(__name__)


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


@dataclass
class RepoReference:
    """
    NB: Is deprecated!
    """

    env_var_name: str
    repo_paths: dict[str, StorageFileReference]

    @property
    def selected(self) -> str:
        import os

        return os.environ[self.env_var_name]

    @property
    def selected_file(self) -> StorageFileReference | None:
        if self.env_var_name in self.repo_paths:
            return self.repo_paths.get(self.env_var_name)
        return self.repo_paths.get(self.selected)

    def feature_server(self, source: FeatureSource) -> FastAPI | FeatureSource:
        import os

        if os.environ.get('ALADDIN_ENABLE_SERVER', 'False').lower() == 'false':
            return source

        from aligned.server import FastAPIServer

        if not (selected_file := self.selected_file):
            raise ValueError('No selected file to serve features from')

        try:
            feature_store = asyncio.get_event_loop().run_until_complete(selected_file.feature_store())
        except RuntimeError:
            import nest_asyncio

            nest_asyncio.apply()
            feature_store = asyncio.new_event_loop().run_until_complete(selected_file.feature_store())

        return FastAPIServer.app(feature_store)

    @staticmethod
    def reference_object(repo: Path, file: Path, object: str) -> RepoReference:
        from aligned.compiler.repo_reader import import_module, path_to_py_module
        from aligned.sources.local import StorageFileReference

        module_path = path_to_py_module(file, repo)

        try:
            module = import_module(module_path)
            obj = getattr(module, object)
            if isinstance(obj, StorageFileReference):
                return RepoReference(env_var_name='const', repo_paths={'const': obj})
            raise ValueError('No reference found')
        except AttributeError:
            raise ValueError('No reference found')
        except ModuleNotFoundError:
            raise ValueError('No reference found')


class FeatureServer:
    @staticmethod
    def from_reference(
        reference: StorageFileReference, online_source: FeatureSource | FeatureSourceFactory | None = None
    ) -> FastAPI | None:
        """Creates a feature server
        This can process and serve features for both models and feature views

        ```python
        redis = RedisConfig.localhost()
        server = FeatureSever.from_reference(
            FileSource.from_json("./feature-store.json"),
            online_source=redis
        )
        ```

        You can then run `aligned serve path_to_server_instance:server`.

        Args:
            reference (StorageFileReference): The location to the feature repository
            online_source (OnlineSource | None, optional): The online source to use.
                Defaults to None meaning the batch source.

        Returns:
            FastAPI: A FastAPI instance that contains paths for fetching features
        """
        import os

        if os.environ.get('ALADDIN_ENABLE_SERVER', 'False').lower() == 'false':
            return None

        from aligned.server import FastAPIServer

        try:
            feature_store = asyncio.get_event_loop().run_until_complete(reference.feature_store())
        except RuntimeError:
            import nest_asyncio

            nest_asyncio.apply()
            feature_store = asyncio.new_event_loop().run_until_complete(reference.feature_store())

        return FastAPIServer.app(feature_store.with_source(online_source))


@dataclass
class EnricherReference(Codable):
    module: str
    attribute_name: str
    enricher: Enricher


@dataclass
class RepoMetadata(Codable):
    created_at: datetime
    name: str
    repo_url: str | None = field(default=None)
    github_url: str | None = field(default=None)


@dataclass
class RepoDefinition(Codable):

    metadata: RepoMetadata

    feature_views: set[CompiledFeatureView] = field(default_factory=set)
    combined_feature_views: set[CompiledCombinedFeatureView] = field(default_factory=set)
    models: set[Model] = field(default_factory=set)
    enrichers: list[EnricherReference] = field(default_factory=list)

    def to_dict(self, **kwargs: dict) -> dict:
        for view in self.feature_views:
            assert isinstance(view, CompiledFeatureView)

        for view in self.combined_feature_views:
            assert isinstance(view, CompiledCombinedFeatureView)

        for model in self.models:
            assert isinstance(model, Model)

        for enricher in self.enrichers:
            assert isinstance(enricher, EnricherReference)
        return super().to_dict(**kwargs)

    @staticmethod
    async def from_file(file: StorageFileReference) -> RepoDefinition:
        repo = await file.read()
        return RepoDefinition.from_json(repo)

    @staticmethod
    async def from_reference_at_path(repo_path: str, file_path: str) -> RepoDefinition:
        from aligned.compiler.repo_reader import RepoReader

        dir_path = Path.cwd() if repo_path == '.' else Path(repo_path).absolute()
        absolute_file_path = Path(file_path).absolute()

        try:
            reference = RepoReader.reference_from_path(dir_path, absolute_file_path)
            if file := reference.selected_file:
                logger.info(f"Loading repo from configuration '{reference.selected}'")
                return await RepoDefinition.from_file(file)
            else:
                logger.info('Found no configuration')
        except ValueError as error:
            logger.error(f'Error when loadin repo: {error}')

        logger.info('Generating repo definition')
        return await RepoReader.definition_from_path(dir_path)

    @staticmethod
    async def from_path(path: str) -> RepoDefinition:
        from aligned.compiler.repo_reader import RepoReader

        dir_path = Path.cwd() if path == '.' else Path(path).absolute()
        return await RepoReader.definition_from_path(dir_path)

    # def add_old_version(self, old_version: "RepoDefinition") -> "RepoDefinition":

    #     views: dict[str, VersionedData[CompiledFeatureView]] = {}
    #     for view in self.feature_views_2:
    #         old_views = [fv for fv in old_version.feature_views_2 if fv.identifier == view.identifier]
    #         if not old_views:
    #             views[view.identifier] = view
    #             continue

    #         old_view = old_views[0]

    #         if old_view.latest == view.latest:
    #             views[view.identifier] = old_view
    #         else:
    #             view[view.identifier] = VersionedData(
    #                   identifier=view.identifier,
    #                   versions=view.versions + old_view.versions
    #               )

    #     self.feature_views_2 = set(views.values())
    #     return self
