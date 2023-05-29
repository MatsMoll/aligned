from dataclasses import dataclass

from aligned.streams.interface import ReadableStream

try:
    from kafka import KafkaConsumer
except ModuleNotFoundError:

    class KafkaConsumer:  # type: ignore
        pass


@dataclass
class KafkaReadableStream(ReadableStream):

    client: KafkaConsumer

    async def read(self, max_records: int = None, max_wait: float = None) -> list[dict]:
        values: list[dict] = []

        raw_values = self.client.poll(timeout_ms=1000)
        for topic in raw_values.values():
            for message in topic:
                values.append(message.value)
        return values
