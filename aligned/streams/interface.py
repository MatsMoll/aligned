class ReadableStream:
    async def read(self, max_records: int = None, max_wait: float = None) -> list[dict]:
        """Reads a stream of data

        ```python
        stream = RedisStream(...)

        records = await stream.read(max_records=10, max_wait=5)
        print(records[0])
        ```
        >>> {"passenger_id": 20, "age": 23, ...}

        Args:
            max_records (int, optional): _description_. Defaults to None.
            max_wait (float, optional): _description_. Defaults to None.

        Returns:
            list[dict]: A list of the records returned
        """
        raise NotImplementedError()


class SinakableStream:
    async def sink(self, records: list[dict]) -> None:
        """Sinkes a record to a stream

        ```python
        stream_source = RedisStream(...)

        await stream_source.sink(records=[
            {"passenger_id": 20, "age": 23, ...}
        ])
        ```

        Args:
            record (dict): The record to sink
        """
        raise NotImplementedError()
