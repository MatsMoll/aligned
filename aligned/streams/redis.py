from dataclasses import dataclass, field

from aligned.schemas.record_coders import PassthroughRecordCoder, RecordCoder
from aligned.streams.interface import ReadableStream, SinakableStream

try:
    from redis.asyncio import Redis  # type: ignore
except ModuleNotFoundError:

    class Redis:  # type: ignore
        async def xread(self, streams: dict[str, str], count: int, block: int) -> list:
            pass

        async def xadd(self, stream: str, record: dict) -> None:
            pass


@dataclass
class RedisStream(ReadableStream, SinakableStream):

    client: Redis
    stream_name: str
    read_timestamp: str = field(default='0-0')
    mappings: dict[str, str] = field(default_factory=dict)
    record_coder: RecordCoder = field(default_factory=lambda: PassthroughRecordCoder())

    async def read(self, max_records: int = None, max_wait: float = None) -> list[dict]:

        stream_values = await self.client.xread(
            streams={self.stream_name: self.read_timestamp}, count=max_records, block=max_wait or 1000
        )

        if not stream_values:
            return []

        # We only listen to one stream, so we will only have one element
        # The first element is the stream name, so we can discard that
        _, values = stream_values[0]
        self.read_timestamp = values[-1][0]

        # We only care about the record, so discarding all ids
        return self.record_coder.decode([record for _, record in values])

    async def sink(self, records: list[dict]) -> None:
        for record in self.record_coder.encode(records):
            await self.client.xadd(self.stream_name, record)
