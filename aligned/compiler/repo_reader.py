import logging
from datetime import datetime
from importlib import import_module
from inspect import getmro, isclass
from typing import Any
from aligned.compiler.model import ModelContractWrapper

from aligned.feature_view.feature_view import FeatureViewWrapper
from aligned.schemas.repo_definition import RepoDefinition, RepoMetadata, RepoReference
from pathlib import Path


logger = logging.getLogger(__name__)


def imports_for(file_path: Path) -> set[str]:
    try:
        with open(file_path) as file:
            raw_file = file.read()
            lines = raw_file.split('\n')
            import_lines = {
                line
                for line in lines
                if line.startswith('import ') or (line.startswith('from ') and ' import ' in line)
            }
            import_set: set[str] = set()
            for line in import_lines:
                tokens = line.split(' ')
                index = tokens.index('import')
                imports = tokens[index:]
                import_set = import_set.union({obj.replace(',', '') for obj in imports})
            return import_set
    except UnicodeDecodeError:
        logger.error(f'Unable to read file: {file_path.as_uri()}')
        return set()


def super_classes_in(obj: Any) -> set[str]:
    super_class_names: set[str] = set()
    if not isclass(obj):
        return set()
    classes = getmro(obj)
    for cl in classes:
        s = str(cl).replace("'", '').replace('>', '')
        super_class_names.add(s.split('.')[-1])

    # Remove the class name. Only store the superclasses
    # Otherwise it might init a abstract class and crash
    s = str(obj).replace("'", '').replace('>', '')
    super_class_names.remove(s.split('.')[-1])
    return super_class_names


def find_files(repo_path: Path, ignore_path: Path | None = None, file_extension: str = 'py') -> list[Path]:
    files = {
        path.resolve()
        for path in repo_path.resolve().glob(f'**/*.{file_extension}')
        if path.is_file()
        and '__init__.py' != path.name
        and not any(part.startswith('.') for part in path.parts)
    }
    if ignore_path:
        ignore_files = {
            path.resolve()
            for path in ignore_path.glob(f'**/*.{file_extension}')
            if path.is_file()
            and '__init__.py' != path.name
            and not any(part.startswith('.') for part in path.parts)
        }
        files -= ignore_files
    return sorted(files)


def path_to_py_module(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root.resolve()))[: -len('.py')].replace('./', '').replace('/', '.')


class RepoReader:
    """
    A class reading a repo, and generates a repo config
    """

    @staticmethod
    async def definition_from_files(files: list[Path], root_path: Path) -> RepoDefinition:

        metadata = RepoMetadata(created_at=datetime.now(), name=root_path.name, github_url=None)
        repo = RepoDefinition(
            metadata=metadata,
            feature_views=set(),
            models=set(),
        )

        for py_file in files:
            imports = imports_for(py_file)
            module_path = path_to_py_module(py_file, root_path)

            if module_path.startswith('aligned') or module_path.startswith('.') or module_path.endswith('__'):
                # Skip no feature defintion modules
                logger.debug(f"Skipping module {module_path}")
                continue

            module = import_module(module_path)

            for attribute in dir(module):
                if attribute in imports:
                    continue

                obj = getattr(module, attribute)

                if isinstance(obj, FeatureViewWrapper):
                    repo.feature_views.add(obj.compile())
                elif isinstance(obj, ModelContractWrapper):
                    repo.models.add(obj.compile())
        return repo

    @staticmethod
    async def definition_from_glob(root_path: Path, glob: str) -> RepoDefinition:
        return await RepoReader.definition_from_files(files=list(root_path.glob(glob)), root_path=root_path)

    @staticmethod
    async def definition_from_path(repo_path: Path, excludes: list[str] | None = None) -> RepoDefinition:

        excluded_files: list[Path] = []
        for exclude in excludes or []:
            excluded_files.extend(repo_path.resolve().glob(exclude))

        return await RepoReader.definition_from_files(
            files=[py_file for py_file in find_files(repo_path) if py_file not in excluded_files],
            root_path=repo_path,
        )

    @staticmethod
    def reference_from_path(repo_path: Path, file: Path) -> RepoReference:

        imports = imports_for(file)

        module_path = path_to_py_module(file, repo_path)
        module = import_module(module_path)

        for attribute in dir(module):
            if attribute in imports:
                continue

            obj = getattr(module, attribute)

            if isinstance(obj, RepoReference):
                return obj
        raise ValueError('No reference found')
