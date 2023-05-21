from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from aligned.compiler.repo_reader import find_files


def generate_snippets():
    root_path = Path().resolve()
    markdown_files = find_files(root_path, file_extension='md')

    source_code_folder = root_path / 'aligned'
    source_code_files = find_files(source_code_folder)

    all_snippets: list[Snippet] = []
    for file in markdown_files:
        all_snippets.extend(generate_snippet_from_markdown_file(file, root_path))

    for file in source_code_files:
        all_snippets.extend(generate_snippet_from_python_file(file, root_path))

    df = pl.DataFrame([asdict(snippet) for snippet in all_snippets]).with_row_count(name='id')
    df.write_csv('snippets.csv', sep=';')


@dataclass
class Snippet:
    source_file: Path
    version_tag: str
    snippet: str


def generate_snippet_from_markdown_file(file: Path, root_path: Path) -> list[Snippet]:
    file_content = file.read_text()
    sections = file_content.split('\n#')
    return [
        Snippet(source_file=file.relative_to(root_path).as_posix(), version_tag='beta', snippet=section)
        for section in sections
    ]


def generate_snippet_from_python_file(file: Path, root_path: Path) -> list[Snippet]:
    file_content = file.read_text()

    dataclass_suffix = '@dataclass\n'
    classes = file_content.split('class ')
    if len(classes) == 1:
        return []
    # The first index will not contain any classes.
    # Therefore, we can remove it
    classes = classes[1:]

    return [
        Snippet(
            source_file=file.relative_to(root_path).as_posix(),
            version_tag='beta',
            snippet=f'class {snippet.removesuffix(dataclass_suffix).strip()}',
        )
        for snippet in classes
    ]


if __name__ == '__main__':
    generate_snippets()
