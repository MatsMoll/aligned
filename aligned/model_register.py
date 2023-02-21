from aligned import AwsS3Config, Model
from aligned.schemas.codable import Codable


class ModelRegistry(Codable):
    """
    A model registry is a place where models are stored and loaded from.

    ```python
    store = await FileSource.from_json("feature-store.json").feature_store()

    weights = await store.model("taxi").load_latest()

    # Or
    weights = await store.model("taxi").load(tag=date(2021, 1, 1)))

    # Or
    await store.model("taxi").release(weights, trained_on=dataset)
    ```
    """

    # async def load_latest(self) -> Model:
    #     pass

    # async def load_tag(self, model_tag: str) -> Model:
    #     pass

    # async def release_model(self, model: Model) -> None:
    #     pass

    # async def list_models(self) -> list[str]:
    #     pass


# class FileSystem:

#     async def list_dir(self, dir: str) -> list[str]:
#         pass

#     async def list_dir_with_metadata(self) -> list[tuple[str, dict]]:
#         # The metadata needs to be semi structured I think
#         pass

#     async def read_file(self, path: str) -> bytes:
#         pass

#     async def write_to(self, path: str, file: bytes) -> None:
#         pass


# class LocalFileSystem(FileSystem):
#     # Use the Path info
#     pass

# class AwsS3FileSystem(FileSystem):
#     # Use the libs that I already have
#     # Will need a Aws Config object tho
#     pass


class ModelDecoder(Codable):
    async def encode(self, model: Model) -> bytes:
        pass

    async def decode(self, model: bytes) -> Model:
        pass


class SklearnModelCoder(ModelDecoder):
    # Using dill which is stright forward
    name = 'sklearn_coder'


# class Dataset:
#     pass


# class DatasetRegistry:

#     async def load_latest(self) -> Dataset:
#         pass

#     async def load_tag(self, dataset_tag: str) -> Dataset:
#         pass

#     async def release_dataset(self, dataset: Dataset) -> None:
#         pass

#     async def list_datasets(self) -> list[str]:
#         pass

# class AwsDatasetRegistry(DatasetRegistry):
#     config: AwsS3Config
#     name = "aws_dataset_registry"

#     # This will also need to take into account different versions.
#     # This means that maybe some metadata is needed to be stored. Aka, a custom format :')
#     # Such a format would not be hard, but it is a slight pain.
#     # However, I need to know which model is created with this dataset (if any)
#     # I need to konw which data is traind, tested, and potentiall validated on
#     # I also need to know which feature store is used. Or rather which features, and feature version are expected


# class AwsModelRegistry(ModelRegistry):
#     coder: ModelDecoder
#     config: AwsS3Config
#     name = "aws_model_registry"

#     @property
#     def file_system(self) -> AwsS3FileSystem:
#         pass
