from dataclasses import dataclass

from aligned.data_source.stream_data_source import StreamDataSource
from aligned.streams.kafka import KafkaReadableStream

try:
    from kafka import KafkaConsumer, TopicPartition
except ModuleNotFoundError:

    class KafkaConsumer:  # type: ignore
        pass


@dataclass
class KafkaTopicConfig(StreamDataSource):

    topic_name: str
    env_key: str

    name: str = 'kafka'

    def kafka_consumer(self) -> KafkaConsumer:
        from os import environ

        if self.env_key not in environ:
            raise ValueError(f'Missing environment key {self.env_key}. Can not connect to Kafka source.')
        consumer = KafkaConsumer(
            bootstrap_servers=[environ[self.env_key]],
        )
        consumer.assign([TopicPartition(self.topic_name, 0)])
        return consumer

    def consumer(self, from_timestamp: str | None = None) -> KafkaReadableStream:
        return KafkaReadableStream(self.kafka_consumer())


@dataclass
class KafkaConfig:

    env_key: str

    def topic(self, topic: str) -> KafkaTopicConfig:
        return KafkaTopicConfig(topic, self.env_key)
