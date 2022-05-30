from dataclasses import dataclass
from http import client
from pathlib import Path
from aioaws.s3 import S3Client

from aladdin.storage import Storage
from aladdin.s3.config import AwsS3Config

@dataclass
class AwsS3Storage(Storage):

    config: AwsS3Config

    async def read(self, path: str) -> bytes:
        from httpx import AsyncClient
        async with AsyncClient() as client:
            s3_client = S3Client(client, self)
            url = s3_client.signed_download_url(path)
            response = await client.get(url)
            return response.content

    async def write(self, path: str, content: bytes) -> None:
        from httpx import AsyncClient
        async with AsyncClient() as client:
            s3_client = S3Client(client, self)
            await s3_client.upload(path, content)

@dataclass
class FileStorage(Storage):

    async def read(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write(self, path: str, content: bytes) -> None:
        return Path(path).write_bytes(content)

@dataclass
class HttpStorage(Storage):

    async def read(self, path: str) -> bytes:
        if not (path.startswith("http://") or path.startswith("https://")):
            raise ValueError("Invalid url")

        from httpx import AsyncClient
        async with AsyncClient() as client:
            response = await client.get(path)
            response.raise_for_status()
            return response.content

    async def write(self, path: str, content: bytes) -> None:
        raise NotImplementedError()
        # return Path(path).write_bytes(content)