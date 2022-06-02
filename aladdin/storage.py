from abc import ABC


class Storage(ABC):
    async def read(self, path: str) -> bytes:
        raise NotImplementedError()

    async def write(self, path: str, content: bytes) -> None:
        raise NotImplementedError()
