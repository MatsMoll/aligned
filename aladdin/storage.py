from abc import ABC


class Storage(ABC):

    async def read(self, path: str) -> bytes:
        raise NotImplementedError()

    async def write(self, path: str, content: bytes) -> None:
        raise NotImplementedError()

    
class LocalFileStorage(Storage):

    def __init__(self, path: str):
        self.path = path

    async def read(self, path: str) -> bytes:
        with open(path, 'rb') as f:
            return f.read()

    async def write(self, path: str, content: bytes) -> None:
        with open(path, 'wb') as f:
            f.write(content)