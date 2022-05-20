from redis.asyncio import Redis
from dataclasses import dataclass

@dataclass
class RedisConfig:

    env_var: str
    _redis: Redis | None = None

    @property
    def url(self) -> str:
        import os
        return os.environ[self.env_var]

    @staticmethod
    def from_url(url: str) -> "RedisConfig":
        import os
        os.environ["REDIS_URL"] = url
        return RedisConfig(env_var="REDIS_URL")

    @staticmethod
    def localhost() -> "RedisConfig":
        import os
        os.environ["REDIS_URL"] = "redis://localhost:6379"
        return RedisConfig(url_env_var="REDIS_URL")

    def redis(self) -> "Redis":
        if self._redis:
            return self._redis
        self._redis = Redis.from_url(self.url)
        return self._redis