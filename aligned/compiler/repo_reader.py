import logging
from importlib import import_module
from inspect import getmro, isclass
from pathlib import Path
from typing import Any

from aligned.enricher import Enricher
from aligned.model import ModelService
from aligned.online_source import BatchOnlineSource, OnlineSource
from aligned.schemas.repo_definition import EnricherReference, RepoDefinition, RepoReference

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


def python_files(repo_path: Path, ignore_path: Path | None = None) -> list[Path]:
    files = {
        path.resolve()
        for path in repo_path.resolve().glob('**/*.py')
        if path.is_file() and '__init__.py' != path.name
    }
    if ignore_path:
        ignore_files = {
            path.resolve()
            for path in ignore_path.glob('**/*.py')
            if path.is_file() and '__init__.py' != path.name
        }
        files -= ignore_files
    return sorted(files)


def path_to_py_module(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root))[: -len('.py')].replace('./', '').replace('/', '.')


class RepoReader:
    """
    A class reading a repo, and generates a repo config
    """

    @staticmethod
    async def definition_from_path(repo_path: Path) -> RepoDefinition:
        repo = RepoDefinition(
            feature_views=set(),
            combined_feature_views=set(),
            models={},
            online_source=BatchOnlineSource(),
            enrichers=[],
        )

        feature_view_names: dict[str, str] = {}

        for py_file in python_files(repo_path):
            imports = imports_for(py_file)

            module_path = path_to_py_module(py_file, repo_path)
            if (
                module_path.startswith('aladdin')
                or module_path.startswith('.')
                or module_path.startswith('heroku')
                or module_path.endswith('__')
            ):
                # Skip aladdin modules
                continue

            module = import_module(module_path)

            for attribute in dir(module):
                if attribute in imports:
                    continue

                obj = getattr(module, attribute)

                if isinstance(obj, ModelService):
                    model_name = obj._name or attribute
                    repo.models[model_name] = obj.feature_refs
                elif isinstance(obj, Enricher):
                    repo.enrichers.append(
                        EnricherReference(module=module_path, attribute_name=attribute, enricher=obj)
                    )
                elif isinstance(obj, OnlineSource):
                    repo.online_source = obj
                else:
                    classes = super_classes_in(obj)
                    if 'FeatureView' in classes:
                        fv = await obj.compile()
                        if fv.name in feature_view_names:
                            raise Exception(
                                (
                                    f'Duplicate feature view names: {fv.name},',
                                    f' in {py_file}, and {feature_view_names[fv.name]}',
                                )
                            )
                        feature_view_names[fv.name] = py_file.as_posix()
                        repo.feature_views.add(fv)
                    elif 'CombinedFeatureView' in classes:
                        fv = await obj.compile()
                        if fv.name in feature_view_names:
                            raise Exception(
                                (
                                    f'Duplicate feature view names: {fv.name},',
                                    f' in {py_file}, and {feature_view_names[fv.name]}',
                                )
                            )
                        feature_view_names[fv.name] = py_file.as_posix()
                        repo.combined_feature_views.add(fv)
        return repo

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
