from __future__ import annotations

from aligned.schemas.codable import Codable
from aligned.config_value import ConfigValue

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass


class AzureCreds(Codable):
    account_id: ConfigValue
    tenant_id: ConfigValue
    client_id: ConfigValue
    client_secret: ConfigValue
    account_name: ConfigValue

    def storage_options(self) -> dict[str, str]:
        return {
            "account_id": self.account_id.read(),
            "tenant_id": self.tenant_id.read(),
            "client_id": self.client_id.read(),
            "client_secret": self.client_secret.read(),
            "account_name": self.account_name.read(),
        }


class AwsCreds:
    pass


class Credentials(Codable):
    azure: AzureCreds | None = None
    aws: AwsCreds | None = None

    def file_system(self) -> str:
        if self.azure:
            return "az://"
        elif self.aws:
            return "s3://"
        else:
            return "file://"

    def storage_options(self) -> dict[str, str] | None:
        if self.azure:
            return self.storage_options()
        return None
