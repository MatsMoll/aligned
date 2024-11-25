from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from aioaws.s3 import S3Client
except ModuleNotFoundError:

    class S3Client:  # type: ignore[no-redef]
        pass


from aligned.storage import Storage

if TYPE_CHECKING:
    from aligned.sources.s3 import AwsS3Config


@dataclass
class AwsS3Storage(Storage):

    config: AwsS3Config
    timeout: int = field(default=60)

    async def read(self, path: str) -> bytes:
        from httpx import AsyncClient

        async with AsyncClient(timeout=self.timeout) as client:
            s3_client = S3Client(client, self.config.s3_config)
            url = s3_client.signed_download_url(path)
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    async def write(self, path: str, content: bytes) -> None:
        from httpx import AsyncClient

        async with AsyncClient(timeout=self.timeout) as client:
            s3_client = S3Client(client, self.config.s3_config)
            await s3_client.upload(path, content)


@dataclass
class FileStorage(Storage):
    async def read(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write(self, path: str, content: bytes) -> None:
        lib_path = Path(path)
        lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib_path.write_bytes(content)


@dataclass
class HttpStorage(Storage):
    async def read(self, path: str) -> bytes:
        if not (path.startswith('http://') or path.startswith('https://')):
            raise ValueError('Invalid url')

        from httpx import AsyncClient

        async with AsyncClient() as client:
            response = await client.get(path)
            response.raise_for_status()
            return response.content

    async def write(self, path: str, content: bytes) -> None:
        raise NotImplementedError()
        # return Path(path).write_bytes(content)
