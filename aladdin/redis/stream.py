from dataclasses import dataclass

from redis.asyncio import Redis  # type: ignore


@dataclass
class RedisStream:

    client: Redis

    async def read_from_timestamp(
        self, streams: dict[str, str], count: int | None = None, timeout: int | None = 5000
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        """Read different streams from a given timestamp.

        Args:
            streams (dict[str, str]): The streams, and the last id to read from
            count (int | None, optional): The max number of values to return.
            Defaults to None.
            timeout (int | None, optional): The amount of milliseconds to block for a read.
            Defaults to 5000, aka 5 sec.

        Returns:
            list[tuple[str, list[tuple[str, dict[str, str]]]]]: The returned streams with their values
        """
        return await self.client.xread(streams, count=count, block=timeout)

    async def read_newest(
        self, streams: list[str], count: int | None = None, timeout: int | None = 5000
    ) -> list[tuple[str, list[tuple[str, dict[str, str]]]]]:
        """Read the values that gets pushed to the stream from now.

        Args:
            streams (list[str]): The streams to read from
            count (int | None, optional): The max number of values to return. Defaults to None.
            timeout (int | None, optional): The amount of milliseconds to block for a read.
             Defaults to 5000, aka 5 sec.

        Returns:
            list[tuple[str, list[tuple[str, dict[str, str]]]]]: The streams with their values
        """
        return await self.read_from_timestamp(
            {stream: '$' for stream in streams}, count=count, timeout=timeout
        )

    # The following methods is used to support a fan out.
    # this is not supported yet

    # async def create_consumer_group(self, group: str) -> None:
    #     await self.client.xgroup_create(self.channel, group)

    # async def read_group(self, group: str, consumer_name: str) -> list[Any]:
    #     return await self.client.xreadgroup(group, consumer_name, {self.channel: ">"})

    # async def acknowledge_message_for(self, group: str, messages: list[str]) -> None:
    #     await self.client.xack(self.channel, group, *messages)
