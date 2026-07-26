from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from aligned.schemas.codable import Codable
from aligned.config_value import ConfigValue, EnvironmentValue, PathResolver

if TYPE_CHECKING:
    from aligned.sources.local import ParquetConfig, DateFormatter, ParquetFileSource


@dataclass
class S3Config(Codable):
    access_key: ConfigValue = field(
        default_factory=lambda: EnvironmentValue("AWS_ACCESS_KEY_ID")
    )
    secret_token: ConfigValue = field(
        default_factory=lambda: EnvironmentValue("AWS_SECRET_ACCESS_KEY")
    )
    region: ConfigValue = field(default_factory=lambda: EnvironmentValue("AWS_REGION"))

    bucket: ConfigValue | None = field(default=None)
    endpoint: ConfigValue | None = field(default=None)

    def storage_options(self) -> dict[str, str]:
        vals = {
            "aws_access_key_id": self.access_key.read(),
            "aws_secret_access_key": self.secret_token.read(),
            "aws_region": self.region.read(),
        }
        if self.bucket:
            vals["aws_bucket"] = self.bucket.read()
        if self.endpoint:
            vals["aws_endpoint"] = self.endpoint.read()
        return vals

    def parquet_at(
        self,
        path: str,
        mapping_keys: dict[str, str] | None = None,
        config: ParquetConfig | None = None,
        date_formatter: DateFormatter | None = None,
    ) -> ParquetFileSource:
        from aligned.sources.local import (
            ParquetFileSource,
            ParquetConfig,
            DateFormatter,
        )

        if "/" not in path:
            # Need to add a base folder if not existing
            path = f"default/{path}"

        return ParquetFileSource(
            PathResolver.from_value(path),
            mapping_keys=mapping_keys or {},
            config=config or ParquetConfig(),
            date_formatter=date_formatter or DateFormatter.noop(),
            s3_config=self,
        )
