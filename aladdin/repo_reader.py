from pathlib import Path
from importlib import import_module
from typing import Any
from inspect import isclass, getmro
from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.repo_definition import RepoDefinition

def imports_for(file_path: Path) -> set[str]:
    with open(file_path, "r") as file:
        raw_file = file.read()
        lines = raw_file.split("\n")
        import_lines = { line for line in lines if line.startswith("import ") or (line.startswith("from ") and " import " in line) }
        import_set = set()
        for line in import_lines:
            tokens = line.split(" ")
            index = tokens.index("import")
            imports = tokens[index:]
            import_set = import_set.union({obj.replace(",", "") for obj in imports})
        return import_set

def super_classes_in(obj: Any) -> set[str]:
    super_class_names: set[str] = set()
    if isclass(obj) == False:
        return set()
    classes = getmro(obj)
    for cl in classes:
        s = str(cl).replace('\'', '').replace('>', '')
        super_class_names.add(s.split('.')[-1])
    
    # Remove the class name. Only store the superclasses
    # Otherwise it might init a abstract class and crash
    s = str(obj).replace('\'', '').replace('>', '')
    super_class_names.remove(s.split('.')[-1])
    return super_class_names

def python_files(repo_path: Path, ignore_path: Path | None = None) -> set[Path]:
    files = {
        path.resolve()
        for path in repo_path.glob("**/*.py")
        if path.is_file() and "__init__.py" != path.name
    }
    if ignore_path:
        ignore_files = {
            path.resolve()
            for path in ignore_path.glob("**/*.py")
            if path.is_file() and "__init__.py" != path.name
        }
        files -= ignore_files
    return sorted(files)

def path_to_py_module(path: Path, repo_root: Path) -> str:
    return (
        str(path.relative_to(repo_root))[: -len(".py")]
        .replace("./", "")
        .replace("/", ".")
    )

class RepoReader:
    """
    A class reading a repo, and generates a repo config
    """

    @staticmethod
    def from_path(repo_path: Path) -> "RepoDefinition":
        repo = RepoDefinition(set(), dict())

        for py_file in python_files(repo_path):
            imports = imports_for(py_file)

            module_path = path_to_py_module(py_file, repo_path)
            if module_path.startswith("aladdin"):
                continue
            module = import_module(module_path)
            
            for attribute in dir(module):
                if attribute in imports:
                    continue

                obj = getattr(module, attribute)

                # if isinstance(obj, FeatureService):
                #     repo.services.add(obj)
                # else:
                classes = super_classes_in(obj)
                if "FeatureView" in classes:
                    repo.feature_views.add(obj.compile())
                # elif "OnDemandView" in classes:
                #     od_view: CompiledOnDemandView = obj.compile(module_path)
                #     repo.on_demand_views.add(od_view)
        return repo