class Storage:
    async def read(self, path: str) -> bytes:
        raise NotImplementedError()

    async def write(self, path: str, content: bytes | bytearray) -> None:
        raise NotImplementedError()
