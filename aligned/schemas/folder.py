from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable

if TYPE_CHECKING:
    from aligned.sources.local import StorageFileReference


class FolderFactory:

    supported_folders: dict[str, type[Folder]]

    _shared: FolderFactory | None = None

    def __init__(self) -> None:
        self.supported_folders = {folder_type.name: folder_type for folder_type in Folder.__subclasses__()}

    @classmethod
    def shared(cls) -> FolderFactory:
        if cls._shared:
            return cls._shared
        cls._shared = FolderFactory()
        return cls._shared


class Folder(Codable, SerializableType):

    name: str

    def file_at(self, path: Path) -> StorageFileReference:
        raise NotImplementedError()

    def _serialize(self) -> dict:
        assert self.name in FolderFactory.shared().supported_folders, f'Unknown type_name: {self.name}'
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> Folder:
        name_type = value['name']
        if name_type not in FolderFactory.shared().supported_folders:
            raise ValueError(
                f"Unknown batch data source id: '{name_type}'.\nRemember to add the"
                ' data source to the FolderFactory.supported_folders if'
                ' it is a custom type.'
                f' Have access to the following types: {FolderFactory.shared().supported_folders.keys()}'
            )
        del value['name']
        data_class = FolderFactory.shared().supported_folders[name_type]
        return data_class.from_dict(value)
