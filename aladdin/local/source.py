from dataclasses import dataclass
from aladdin.data_source.batch_data_source import BatchDataSource, ColumnFeatureMappable

@dataclass
class LocalFileSource(BatchDataSource, ColumnFeatureMappable):

    path: str
    mapping_keys: dict[str, str]

    type_name: str = "local_file"

    def job_group_key(self) -> str:
        return f"{self.type_name}/{self.path}"

    def __hash__(self) -> int:
        return hash(self.job_group_key())

