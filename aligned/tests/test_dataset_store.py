import json
from aligned import FileSource, AzureBlobConfig
from aligned.compiler.model import resolve_dataset_store
from aligned.config_value import EnvironmentValue, LiteralValue, NothingValue
from aligned.schemas.folder import DatasetStore


def test_dataset_sources_are_codable() -> None:
    data_sources = [
        FileSource.json_at("test.json"),
        AzureBlobConfig(
            LiteralValue("constante"),
            EnvironmentValue("ENV_VAR"),
            NothingValue(),
            NothingValue(),
            NothingValue(),
        ).json_at("test.json"),
    ]

    for source in data_sources:
        data_set_store = resolve_dataset_store(source)

        json_data = data_set_store.to_json()

        encoded = DatasetStore._deserialize(json.loads(json_data))

        assert encoded == data_set_store
