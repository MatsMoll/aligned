from dataclasses import dataclass
from io import StringIO

from pandas import DataFrame

from aladdin.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable
from aladdin.feature_store import FeatureStore
from aladdin.repo_definition import RepoDefinition
from aladdin.s3.storage import FileStorage, HttpStorage
from aladdin.storage import Storage


class FileReference:
    async def read(self) -> bytes:
        raise NotImplementedError()

    async def write(self, content: bytes) -> None:
        raise NotImplementedError()

    async def feature_store(self) -> FeatureStore:
        file = await self.read()
        return FeatureStore.from_definition(RepoDefinition.from_json(file))


@dataclass
class FileSource(BatchDataSource, ColumnFeatureMappable, FileReference):

    path: str
    mapping_keys: dict[str, str]

    type_name: str = 'local_file'

    @property
    def storage(self) -> Storage:
        if self.path.startswith('http'):
            return HttpStorage()
        else:
            return FileStorage()

    def job_group_key(self) -> str:
        return f'{self.type_name}/{self.path}'

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    @staticmethod
    def from_path(path: str) -> 'FileSource':
        return FileSource(path=path, mapping_keys={})

    async def read(self) -> bytes:
        return await self.storage.read(self.path)

    async def write(self, content: bytes) -> None:
        return await self.storage.write(self.path, content)


@dataclass
class LiteralReference(FileReference):

    file: DataFrame

    async def read(self) -> bytes:
        file = StringIO()
        self.file.to_csv(file, index=False)
        file.seek(0)
        return file.read().encode('utf-8')

    async def write(self, content: bytes) -> None:
        raise NotImplementedError()
