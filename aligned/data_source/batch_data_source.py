from __future__ import annotations
from contextlib import suppress
from copy import copy
from datetime import datetime

from typing import (
    TYPE_CHECKING,
    Awaitable,
    Iterable,
    Literal,
    TypeVar,
    Any,
    Callable,
    Coroutine,
)
from dataclasses import dataclass, field

from mashumaro.types import SerializableType
from aligned.config_value import ConfigValue
from aligned.data_file import DataFileReference

from aligned.retrieval_job import FilterRepresentable
from aligned.schemas.codable import Codable
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import (
    Feature,
    FeatureLocation,
    FeatureType,
)
from aligned.request.retrieval_request import RequestResult, RetrievalRequest
from aligned.compiler.feature_factory import FeatureFactory
from polars._typing import TimeUnit
import polars as pl

import logging

from aligned.schemas.transformation import Expression


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aligned.retrieval_job import RetrievalJob
    from aligned.schemas.feature_view import CompiledFeatureView

T = TypeVar("T")


class AsBatchSource:
    def as_source(self) -> CodableBatchDataSource:
        raise NotImplementedError(type(self))


class BatchDataSourceFactory:
    supported_data_sources: dict[str, type[CodableBatchDataSource]]

    _shared: BatchDataSourceFactory | None = None

    def __init__(self) -> None:
        from aligned.sources.local import (
            CsvFileSource,
            ParquetFileSource,
            DeltaFileSource,
            PartitionedParquetFileSource,
        )
        from aligned.schemas.feature_view import FeatureViewReferenceSource
        from aligned.schemas.model import ModelSource
        from aligned.sources.databricks import (
            UCTableSource,
            UCFeatureTableSource,
            UCSqlSource,
        )

        from aligned.sources.psql import PostgreSQLDataSource
        from aligned.sources.redshift import RedshiftSQLDataSource
        from aligned.sources.s3 import AwsS3CsvDataSource, AwsS3ParquetDataSource
        from aligned.sources.azure_blob_storage import (
            AzureBlobCsvDataSource,
            AzureBlobParquetDataSource,
            AzureBlobPartitionedParquetDataSource,
        )
        from aligned.sources.lancedb import LanceDbTable

        source_types = [
            PostgreSQLDataSource,
            UCTableSource,
            UCFeatureTableSource,
            UCSqlSource,
            # File Sources
            CsvFileSource,
            DeltaFileSource,
            ParquetFileSource,
            PartitionedParquetFileSource,
            # Aws Sources
            AwsS3CsvDataSource,
            AwsS3ParquetDataSource,
            RedshiftSQLDataSource,
            # Azure Sources
            AzureBlobCsvDataSource,
            AzureBlobParquetDataSource,
            AzureBlobPartitionedParquetDataSource,
            # LanceDB
            LanceDbTable,
            # Aligned Related Sources
            JoinDataSource,
            JoinAsofDataSource,
            FilteredDataSource,
            FeatureViewReferenceSource,
            CustomMethodDataSource,
            ModelSource,
            StackSource,
            LoadedAtSource,
            TransformSource,
            # InMemorySource,
        ]

        self.supported_data_sources = {
            source.type_name: source for source in source_types
        }

    @classmethod
    def shared(cls) -> BatchDataSourceFactory:
        if cls._shared:
            return cls._shared
        cls._shared = BatchDataSourceFactory()
        return cls._shared


class BatchSourceModification:
    source: CodableBatchDataSource

    def wrap_job(self, job: RetrievalJob) -> RetrievalJob:
        raise NotImplementedError()


class BatchDataSource:
    def job_group_key(self) -> str:
        """
        A key defining which sources can be grouped together in one request.
        """
        raise NotImplementedError(type(self))

    def source_id(self) -> str:
        """
        An id that identifies a source from others.
        """
        return self.job_group_key()

    def needed_configs(self) -> list[ConfigValue]:
        return []

    def with_view(self: T, view: CompiledFeatureView) -> T:
        return self

    def __hash__(self) -> int:
        return hash(self.job_group_key())

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        if isinstance(self, BatchSourceModification):
            return self.wrap_job(self.source.all_data(request, limit))

        if isinstance(self, DataFileReference):
            from aligned.local.job import FileFullJob

            return FileFullJob(self, request=request, _limit=limit)

        raise NotImplementedError(type(self))

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        if isinstance(self, BatchSourceModification):
            return self.wrap_job(
                self.source.all_between_dates(request, start_date, end_date)
            )

        if isinstance(self, DataFileReference):
            from aligned.local.job import FileDateJob

            return FileDateJob(
                self, request=request, start_date=start_date, end_date=end_date
            )

        raise NotImplementedError(type(self))

    @classmethod
    def multi_source_features_for(
        cls: type[T], facts: RetrievalJob, requests: list[tuple[T, RetrievalRequest]]
    ) -> RetrievalJob:
        sources = {
            source.job_group_key()
            for source, _ in requests
            if isinstance(source, CodableBatchDataSource)
        }
        if len(sources) != 1:
            raise NotImplementedError(
                f"Type: {cls} have not implemented how to load fact data with multiple sources."
            )

        source, _ = requests[0]
        if isinstance(source, BatchSourceModification):
            return source.wrap_job(
                type(source.source).multi_source_features_for(facts, requests)  # type: ignore
            )
        elif isinstance(source, DataFileReference):
            from aligned.local.job import FileFactualJob

            return FileFactualJob(source, [request for _, request in requests], facts)
        else:
            raise NotImplementedError(
                f"Type: {cls} have not implemented how to load fact data"
            )

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        return type(self).multi_source_features_for(facts, [(self, request)])

    async def schema(self) -> dict[str, FeatureType]:
        """Returns the schema for the data source

        ```python
        source = FileSource.parquet_at('test_data/titanic.parquet')
        schema = await source.schema()
        >>> {'passenger_id': FeatureType(name='int64'), ...}
        ```

        Returns:
            dict[str, FeatureType]: A dictionary containing the column name and the feature type
        """
        if isinstance(self, BatchSourceModification):
            return await self.source.schema()

        raise NotImplementedError(f"`schema()` is not implemented for {type(self)}.")

    def all_columns(self, limit: int | None = None) -> RetrievalJob:
        return self.all(RequestResult(set(), set(), None), limit=limit)

    def all(self, result: RequestResult, limit: int | None = None) -> RetrievalJob:
        return self.all_data(
            result.as_retrieval_request(
                "read_all", location=FeatureLocation.feature_view("read_all")
            ),
            limit=limit,
        )

    async def feature_view_code(self, view_name: str) -> str:
        """Setup the code needed to represent the data source as a feature view

        ```python
        FileSource.parquet("my_path.parquet").feature_view_code(view_name="my_view")

        >>> \"\"\"from aligned import FeatureView, String, Int64, Float

        class MyView(FeatureView):

            metadata = FeatureView.metadata_with(
                name="Embarked",
                description="some description",
                batch_source=FileSource.parquest("my_path.parquet")
                stream_source=None,
            )

            Passenger_id = Int64()
            Survived = Int64()
            Pclass = Int64()
            Name = String()
            Sex = String()
            Age = Float()
            Sibsp = Int64()
            Parch = Int64()
            Ticket = String()
            Fare = Float()
            Cabin = String()
            Embarked = String()\"\"\"
        ```

        Returns:
            str: The code needed to setup a basic feature view
        """
        from aligned.feature_view.feature_view import FeatureView

        schema = await self.schema()
        feature_types = {
            name: feature_type.feature_factory for name, feature_type in schema.items()
        }
        return FeatureView.feature_view_code_template(
            feature_types, f"{self}", view_name
        )

    async def freshness(self, feature: Feature) -> datetime | None:
        """
        my_table_freshenss = await (PostgreSQLConfig("DB_URL")
            .table("my_table")
            .freshness()
        )
        """
        from aligned.data_file import DataFileReference
        from aligned.sources.local import data_file_freshness

        if isinstance(self, DataFileReference):
            return await data_file_freshness(self, feature.name)

        raise NotImplementedError(f"Freshness is not implemented for {type(self)}.")

    def filter(self, condition: FilterRepresentable) -> CodableBatchDataSource:
        assert isinstance(self, CodableBatchDataSource)
        return FilteredDataSource(self, Expression.from_value(condition))

    def location_id(self) -> set[FeatureLocation]:
        return self.depends_on()

    def depends_on(self) -> set[FeatureLocation]:
        return set()

    def tags(self) -> list[str]:
        if isinstance(self, CodableBatchDataSource):
            return [self.type_name]
        return []

    def with_loaded_at(self) -> CodableBatchDataSource:
        if isinstance(self, CodableBatchDataSource):
            return LoadedAtSource(self)
        raise NotImplementedError(type(self))

    def transform_with_polars(
        self,
        method: Callable[[pl.LazyFrame], Awaitable[pl.LazyFrame]]
        | Callable[[pl.LazyFrame], pl.LazyFrame],
        docker_config: DockerConfig | str | None = None,
    ) -> CodableBatchDataSource:
        codable = CodableFunction.from_function(method)
        assert codable is not None

        return TransformSource(
            self,  # type: ignore
            codable,
            docker_config=DockerConfig(docker_config)
            if isinstance(docker_config, str)
            else docker_config,
        )


class CodableBatchDataSource(Codable, SerializableType, BatchDataSource):
    """
    A definition to where a specific piece of data can be found.
    E.g: A database table, a file, a web service, etc.

    This can thereafter be combined with other BatchDataSources in order to create a rich dataset.
    """

    type_name: str

    @property
    def as_markdown(self) -> str:
        return self.type_name

    def _serialize(self) -> dict:
        from aligned.sources.in_mem_source import InMemorySource

        if self.type_name == InMemorySource.type_name:
            return InMemorySource.to_dict(self)  # type: ignore

        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> CodableBatchDataSource:
        from aligned.sources.in_mem_source import InMemorySource

        name_type = value.get("type_name", "missing type name in source")

        if name_type == InMemorySource.type_name:
            import io

            if "data" in value:
                json_bytes = io.BytesIO(bytes.fromhex(value["data"]))
                return InMemorySource(
                    data=pl.DataFrame.deserialize(json_bytes, format="binary")
                )
            else:
                return InMemorySource.from_dict(value)

        if name_type not in BatchDataSourceFactory.shared().supported_data_sources:
            return UnknownDataSource(type_name=name_type, content=value)

        del value["type_name"]
        data_class = BatchDataSourceFactory.shared().supported_data_sources[name_type]
        return data_class.from_dict(value)


@dataclass
class UnknownDataSource(CodableBatchDataSource):
    type_name: str
    content: dict

    @property
    def as_markdown(self) -> str:
        return f"Unknown source named {self.type_name} with content {self.content}"

    def job_group_key(self) -> str:
        from uuid import uuid4

        return str(uuid4())

    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        return d["content"]

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        raise NotImplementedError(
            f"Missing implementation for source with content {self.content}"
        )

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        raise NotImplementedError(
            f"Missing implementation for source with content {self.content}"
        )

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        raise NotImplementedError(
            f"Missing implementation for source with content {self.content}"
        )


@dataclass
class DockerConfig(Codable):
    image_url: str
    username: ConfigValue | None = field(default=None)
    password: ConfigValue | None = field(default=None)


@dataclass
class FunctionReference(Codable):
    module_location: str
    function_name: str


def make_async(function: Callable):
    from inspect import iscoroutinefunction

    if iscoroutinefunction(function):
        return function

    async def async_func(*args, **kwargs):
        function(*args, **kwargs)

    return async_func


@dataclass
class CodableFunction(Codable):
    dill_bytes: bytes | None = None
    function_ref: FunctionReference | None = None

    @property
    def as_markdown(self) -> str:
        if self.function_ref:
            return f"Function named `{self.function_ref.function_name}` at `{self.function_ref.module_location}`"

        return "Binary encoded method"

    def load_function(self) -> Callable | None:
        if self.function_ref:
            with suppress(ImportError, AttributeError):
                import importlib

                module = importlib.import_module(self.function_ref.module_location)
                return getattr(module, self.function_ref.function_name)

        if self.dill_bytes:
            import dill

            return dill.loads(self.dill_bytes)

        return None

    @staticmethod
    def from_function(function: Callable | None) -> CodableFunction | None:
        import inspect

        if function is None:
            return None

        file = inspect.getmodule(function)

        def dill_function():
            import dill

            return CodableFunction(dill_bytes=dill.dumps(function))

        if ">" in function.__name__ or "<" in function.__name__:
            return dill_function()

        if file is None:
            return dill_function()

        if "aligned" in file.__name__:
            return dill_function()

        return CodableFunction(
            dill_bytes=None,
            function_ref=FunctionReference(
                module_location=file.__name__, function_name=function.__name__
            ),
        )


@dataclass
class TransformSource(CodableBatchDataSource):
    source: CodableBatchDataSource
    generic_method: CodableFunction
    docker_config: DockerConfig | None = None

    type_name: str = "transform_source"

    @property
    def as_markdown(self) -> str:
        markdown = "### Transform Source"

        if self.generic_method:
            markdown += f"\n\n**Generic load**: {self.generic_method.as_markdown}"

        if self.docker_config:
            markdown += f"\n\n**Docker config**: {self.docker_config}"

        return markdown

    def job_group_key(self) -> str:
        from hashlib import sha256

        return sha256(self.generic_method.as_markdown.encode()).hexdigest()

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        method = self.generic_method.load_function()
        assert method
        return self.source.all_data(request, limit).transform_polars(method)  # type: ignore

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        method = self.generic_method.load_function()
        assert method
        return self.source.all_between_dates(
            request, start_date, end_date
        ).transform_polars(method)  # type: ignore

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        method = self.generic_method.load_function()
        assert method
        return self.source.features_for(facts, request).transform_polars(method)  # type: ignore

    @classmethod
    def multi_source_features_for(
        cls: type[T], facts: RetrievalJob, requests: list[tuple[T, RetrievalRequest]]
    ) -> RetrievalJob:
        if len(requests) != 1:
            raise NotImplementedError(
                f"Type: {cls} have not implemented how to load fact data with multiple sources."
            )

        source, request = requests[0]
        return source.features_for(facts, request)  # type: ignore

    def depends_on(self) -> set[FeatureLocation]:
        return self.source.depends_on()


@dataclass
class CustomMethodDataSource(CodableBatchDataSource):
    all_data_method: CodableFunction | None
    all_between_dates_method: CodableFunction | None
    features_for_method: CodableFunction | None

    generic_method: CodableFunction | None

    depends_on_sources: set[FeatureLocation] | None = None
    docker_config: DockerConfig | None = None

    type_name: str = "custom_method"

    @property
    def as_markdown(self) -> str:
        markdown = "### Custom Method Source"

        if self.all_data_method:
            markdown += f"\n\n**All data**: {self.all_data_method.as_markdown}"

        if self.all_between_dates_method:
            markdown += (
                f"\n\n**Between date**: {self.all_between_dates_method.as_markdown}"
            )

        if self.features_for_method:
            markdown += f"\n\n**Features for**: {self.features_for_method.as_markdown}"

        if self.generic_method:
            markdown += f"\n\n**Generic load**: {self.generic_method.as_markdown}"

        if self.docker_config:
            markdown += f"\n\n**Docker config**: {self.docker_config}"

        return markdown

    def job_group_key(self) -> str:
        from hashlib import sha256

        if self.generic_method:
            return sha256(self.generic_method.as_markdown.encode()).hexdigest()

        description = ""
        if self.all_data_method:
            description += self.all_data_method.as_markdown
        if self.all_between_dates_method:
            description += self.all_between_dates_method.as_markdown
        if self.features_for_method:
            description += self.features_for_method.as_markdown

        return f"custom_method-{sha256(description.encode()).hexdigest()}"

    @property
    def all_method(
        self,
    ) -> Callable[[RetrievalRequest, int | None], Coroutine[None, None, pl.LazyFrame]]:
        function = None
        if self.all_data_method:
            function = self.all_data_method.load_function()

        if function is None and self.generic_method:
            sub_function = self.generic_method.load_function()
            assert sub_function

            async def wrapped_function(
                req: RetrievalRequest, limit: int | None
            ) -> pl.LazyFrame:
                return await make_async(sub_function)(req)  # type: ignore

            function = wrapped_function

        return make_async(function or CustomMethodDataSource.default_throw)  # type: ignore

    @property
    def between_method(
        self,
    ) -> Callable[
        [RetrievalRequest, datetime, datetime], Coroutine[None, None, pl.LazyFrame]
    ]:
        function = None
        if self.all_between_dates_method:
            function = self.all_between_dates_method.load_function()

        if function is None and self.generic_method:
            sub_function = self.generic_method.load_function()
            assert sub_function

            async def wrapped_function(
                req: RetrievalRequest, start: datetime, end: datetime
            ) -> pl.LazyFrame:
                return await make_async(sub_function)(req)  # type: ignore

            function = wrapped_function

        return make_async(function or CustomMethodDataSource.default_throw)  # type: ignore

    @property
    def for_method(
        self,
    ) -> Callable[
        [RetrievalJob, RetrievalRequest], Coroutine[None, None, pl.LazyFrame]
    ]:
        function = None
        if self.features_for_method:
            function = self.features_for_method.load_function()

        if function is None and self.generic_method:
            sub_function = self.generic_method.load_function()
            assert sub_function

            async def wrapped_function(
                facts: RetrievalJob, req: RetrievalRequest
            ) -> pl.LazyFrame:
                return await make_async(sub_function)(req)  # type: ignore

            function = wrapped_function

        return make_async(function or CustomMethodDataSource.default_throw)  # type: ignore

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        from aligned.retrieval_job import CustomLazyPolarsJob

        method = self.all_method

        return CustomLazyPolarsJob(
            request=request,
            method=lambda: method(request, limit),
        ).fill_missing_columns()

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        from aligned.retrieval_job import CustomLazyPolarsJob

        method = self.between_method

        return CustomLazyPolarsJob(
            request=request,
            method=lambda: method(request, start_date, end_date),
        ).fill_missing_columns()

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        from aligned.retrieval_job import CustomLazyPolarsJob

        method = self.for_method

        return CustomLazyPolarsJob(
            request=request,
            method=lambda: method(facts, request),
        ).fill_missing_columns()

    @classmethod
    def multi_source_features_for(
        cls: type[T], facts: RetrievalJob, requests: list[tuple[T, RetrievalRequest]]
    ) -> RetrievalJob:
        if len(requests) != 1:
            raise NotImplementedError(
                f"Type: {cls} have not implemented how to load fact data with multiple sources."
            )

        source, request = requests[0]
        return source.features_for(facts, request)  # type: ignore

    @staticmethod
    def from_load(
        method: Callable[[RetrievalRequest], Coroutine[None, None, pl.LazyFrame]],
        depends_on: set[FeatureLocation] | None = None,
        docker_config: DockerConfig | None = None,
    ) -> "CustomMethodDataSource":
        return CustomMethodDataSource.from_methods(
            generic_load=method,
            depends_on_sources=depends_on,
            docker_config=docker_config,
        )

    @staticmethod
    def from_methods(
        all_data: Callable[
            [RetrievalRequest, int | None], Coroutine[None, None, pl.LazyFrame]
        ]
        | None = None,
        all_between_dates: Callable[
            [RetrievalRequest, datetime, datetime], Coroutine[None, None, pl.LazyFrame]
        ]
        | None = None,
        features_for: Callable[
            [RetrievalJob, RetrievalRequest], Coroutine[None, None, pl.LazyFrame]
        ]
        | None = None,
        generic_load: Callable[[RetrievalRequest], Coroutine[None, None, pl.LazyFrame]]
        | None = None,
        depends_on_sources: set[FeatureLocation] | None = None,
        docker_config: DockerConfig | str | None = None,
    ) -> "CustomMethodDataSource":
        if isinstance(docker_config, str):
            docker_config = DockerConfig(image_url=docker_config)

        return CustomMethodDataSource(
            all_data_method=CodableFunction.from_function(all_data),
            all_between_dates_method=CodableFunction.from_function(all_between_dates),
            features_for_method=CodableFunction.from_function(features_for),
            generic_method=CodableFunction.from_function(generic_load),
            depends_on_sources=depends_on_sources,
            docker_config=docker_config,
        )

    @staticmethod
    def default_throw(*args: Any, **kwargs: Any) -> pl.LazyFrame:
        raise NotImplementedError("No method is defined for this data source.")

    def depends_on(self) -> set[FeatureLocation]:
        return self.depends_on_sources or set()


@dataclass
class FilteredDataSource(CodableBatchDataSource):
    source: CodableBatchDataSource
    condition: Expression

    type_name: str = "subset"

    def job_group_key(self) -> str:
        return f"subset/{self.source.job_group_key()}"

    async def schema(self) -> dict[str, FeatureType]:
        return await self.source.schema()

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls: type[FilteredDataSource],
        facts: RetrievalJob,
        requests: list[tuple[FilteredDataSource, RetrievalRequest]],
    ) -> RetrievalJob:
        sources = {
            source.job_group_key()
            for source, _ in requests
            if isinstance(source, CodableBatchDataSource)
        }
        if len(sources) != 1:
            raise NotImplementedError(
                f"Type: {cls} have not implemented how to load fact data with multiple sources."
            )
        source, request = requests[0]
        return source.source.features_for(facts, request).filter(source.condition)

    async def freshness(self, feature: Feature) -> datetime | None:
        return await self.source.freshness(feature)

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        return (
            self.source.all_between_dates(request, start_date, end_date)
            .filter(self.condition)
            .aggregate(request)
            .derive_features([request])
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        if isinstance(self.condition, DerivedFeature):
            request.derived_features.add(self.condition)
            return (
                self.source.all_data(request, limit)
                .aggregate(request)
                .derive_features([request])
                .filter(self.condition)
            )
        elif isinstance(self.condition, Feature):
            request.features.add(self.condition)

        return (
            self.source.all_data(request, limit)
            .filter(self.condition)
            .aggregate(request)
            .derive_features([request])
        )

    def depends_on(self) -> set[FeatureLocation]:
        return self.source.depends_on()


def resolve_keys(
    keys: str | FeatureFactory | list[str] | list[FeatureFactory],
) -> list[str]:
    if isinstance(keys, FeatureFactory):
        return [keys.name]

    if isinstance(keys, str):
        return [keys]

    if isinstance(keys[0], FeatureFactory):
        return [key.name for key in keys]  # type: ignore

    return keys  # type: ignore


def model_prediction_instance_source(
    model: Any,
) -> tuple[CodableBatchDataSource, RetrievalRequest] | Exception:
    from aligned.schemas.feature_view import FeatureViewReferenceSource
    from aligned.compiler.model import ModelContractWrapper

    if not hasattr(model, "__model_wrapper__"):
        return ValueError(
            f"Unable to join {model} as a __view_wrapper__ is needed. Make sure you have used @feature_view"
        )

    wrapper = getattr(model, "__model_wrapper__")
    if not isinstance(wrapper, ModelContractWrapper):
        return ValueError()

    compiled_view = wrapper.as_view()
    if compiled_view is None:
        return ValueError()

    return (
        FeatureViewReferenceSource(
            compiled_view, FeatureLocation.model(compiled_view.name)
        ),
        compiled_view.request_all.needed_requests[0],
    )


def view_wrapper_instance_source(
    view: Any,
) -> tuple[CodableBatchDataSource, RetrievalRequest] | Exception:
    from aligned.feature_view.feature_view import FeatureViewWrapper
    from aligned.schemas.feature_view import FeatureViewReferenceSource

    if not hasattr(view, "__view_wrapper__"):
        return ValueError(
            f"Unable to join {view} as a __view_wrapper__ is needed. Make sure you have used @feature_view"
        )

    wrapper = getattr(view, "__view_wrapper__")
    if not isinstance(wrapper, FeatureViewWrapper):
        return ValueError()

    compiled_view = wrapper.compile()

    return (
        FeatureViewReferenceSource(
            compiled_view, FeatureLocation.feature_view(compiled_view.name)
        ),
        compiled_view.request_all.needed_requests[0],
    )


def join_asof_source(
    source: CodableBatchDataSource,
    left_request: RetrievalRequest,
    view: Any,
    left_on: list[str] | None = None,
    right_on: list[str] | None = None,
) -> JoinAsofDataSource:
    wrapped_source = view_wrapper_instance_source(view)
    if isinstance(wrapped_source, Exception):
        wrapped_source = model_prediction_instance_source(view)

    if isinstance(wrapped_source, Exception):
        raise wrapped_source

    right_source, right_request = wrapped_source

    left_event_timestamp = left_request.event_timestamp
    right_event_timestamp = right_request.event_timestamp

    if left_event_timestamp is None:
        raise ValueError("A left event timestamp is needed, but found none.")
    if right_event_timestamp is None:
        raise ValueError("A right event timestamp is needed, but found none.")

    return JoinAsofDataSource(
        source=source,
        left_request=left_request,
        right_source=right_source,
        right_request=right_request,
        left_event_timestamp=left_event_timestamp.name,
        right_event_timestamp=right_event_timestamp.name,
        left_on=left_on,
        right_on=right_on,
    )


def join_source(
    source: CodableBatchDataSource,
    view: Any,
    on_left: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
    on_right: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
    how: Literal["inner", "left", "outer"] = "inner",
    left_request: RetrievalRequest | None = None,
) -> JoinDataSource:
    from aligned.data_source.batch_data_source import JoinDataSource
    from aligned.feature_view.feature_view import FeatureViewWrapper

    wrapped_source = view_wrapper_instance_source(view)
    if isinstance(wrapped_source, Exception):
        wrapped_source = model_prediction_instance_source(view)

    if isinstance(wrapped_source, Exception):
        raise wrapped_source

    right_source, right_request = wrapped_source

    if on_left is None:
        left_keys = list(right_request.entity_names)
    else:
        left_keys = resolve_keys(on_left)

    if on_right is None:
        right_keys = list(right_request.entity_names)
    else:
        right_keys = resolve_keys(on_right)

    if left_request is None:
        if isinstance(source, JoinDataSource):
            left_request = RetrievalRequest.unsafe_combine(
                [source.left_request, source.right_request]
            )
        elif isinstance(source, FeatureViewWrapper):
            left_request = source.compile().request_all.needed_requests[0]

    if left_request is None:
        raise ValueError(
            "Unable to resolve the left request. Consider adding a `left_request` param."
        )

    return JoinDataSource(
        source=source,
        left_request=left_request,
        right_source=right_source,
        right_request=right_request,
        left_on=left_keys,
        right_on=right_keys,
        method=how,
    )


@dataclass
class JoinAsofDataSource(CodableBatchDataSource):
    source: CodableBatchDataSource
    left_request: RetrievalRequest
    right_source: CodableBatchDataSource
    right_request: RetrievalRequest

    left_event_timestamp: str
    right_event_timestamp: str

    left_on: list[str] | None = None
    right_on: list[str] | None = None

    timestamp_unit: TimeUnit = "us"

    type_name: str = "join_asof"

    async def schema(self) -> dict[str, FeatureType]:
        left_schema = await self.source.schema()
        right_schema = await self.right_source.schema()

        return {**left_schema, **right_schema}

    def job_group_key(self) -> str:
        return f"join/{self.source.job_group_key()}"

    def all_with_limit(self, limit: int | None) -> RetrievalJob:
        right_job = self.right_source.all_data(
            self.right_request, limit=None
        ).derive_features([self.right_request])

        return (
            self.source.all_data(self.left_request, limit=limit)
            .derive_features([self.left_request])
            .join_asof(
                right_job,
                left_event_timestamp=self.left_event_timestamp,
                right_event_timestamp=self.right_event_timestamp,
                left_on=self.left_on,
                right_on=self.right_on,
                timestamp_unit=self.timestamp_unit,
            )
            .fill_missing_columns()
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        right_job = self.right_source.all_data(
            self.right_request, limit=None
        ).derive_features([self.right_request])

        return (
            self.source.all_data(self.left_request, limit=limit)
            .derive_features([self.left_request])
            .join_asof(
                right_job,
                left_event_timestamp=self.left_event_timestamp,
                right_event_timestamp=self.right_event_timestamp,
                left_on=self.left_on,
                right_on=self.right_on,
                timestamp_unit=self.timestamp_unit,
            )
            .aggregate(request)
            .fill_missing_columns()
            .derive_features([request])
        )

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        right_job = self.right_source.all_data(
            self.right_request, limit=None
        ).derive_features([self.right_request])

        return (
            self.source.all_between_dates(self.left_request, start_date, end_date)
            .derive_features([self.left_request])
            .join_asof(
                right_job,
                left_event_timestamp=self.left_event_timestamp,
                right_event_timestamp=self.right_event_timestamp,
                left_on=self.left_on,
                right_on=self.right_on,
            )
            .aggregate(request)
            .fill_missing_columns()
            .derive_features([request])
        )

    async def freshness(self, feature: Feature) -> datetime | None:
        left_freshness = await self.source.freshness(feature)
        right_frehsness = await self.right_source.freshness(feature)

        if left_freshness is None:
            return None

        if right_frehsness is None:
            return None

        return min(left_freshness, right_frehsness)

    def join(
        self,
        view: Any,
        on: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        how: str = "inner",
    ) -> JoinDataSource:
        return join_source(self, view, on, how)

    def join_asof(
        self, view: Any, on: str | FeatureFactory | list[str] | list[FeatureFactory]
    ) -> JoinAsofDataSource:
        left_on = None
        right_on = None
        if on:
            left_on = resolve_keys(on)
            right_on = left_on

        left_request = RetrievalRequest.unsafe_combine(
            [self.left_request, self.right_request]
        )

        return join_asof_source(
            self,
            left_request=left_request,
            view=view,
            left_on=left_on,
            right_on=right_on,
        )

    def depends_on(self) -> set[FeatureLocation]:
        return self.source.depends_on().union(self.right_source.depends_on())


@dataclass
class StackSource(CodableBatchDataSource):
    top: CodableBatchDataSource
    bottom: CodableBatchDataSource

    source_column: str | None = None

    type_name: str = "stack"

    @property
    def source_column_config(self):  # type: ignore
        from aligned.retrieval_job import StackSourceColumn

        if not self.source_column:
            return None

        return StackSourceColumn(
            top_source_name=self.top.source_id(),
            bottom_source_name=self.bottom.source_id(),
            source_column=self.source_column,
        )

    def sub_request(self, request: RetrievalRequest, config) -> RetrievalRequest:  # type: ignore
        return RetrievalRequest(
            name=request.name,
            location=request.location,
            features={
                feature
                for feature in request.features
                if feature.name != config.source_column
            },
            entities=request.entities,
            derived_features={
                feature
                for feature in request.derived_features
                if not any(
                    dep.name == config.source_column for dep in feature.depending_on
                )
            },
            aggregated_features=request.aggregated_features,
            event_timestamp_request=request.event_timestamp_request,
            features_to_include=request.features_to_include - {config.source_column},
        )

    def job_group_key(self) -> str:
        return f"stack/{self.top.job_group_key()}/{self.bottom.job_group_key()}"

    async def schema(self) -> dict[str, FeatureType]:
        top_schema = await self.top.schema()
        bottom_schema = await self.bottom.schema()

        return {**top_schema, **bottom_schema}

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        from aligned.retrieval_job import StackJob

        config = self.source_column_config

        sub_request = request

        if config:
            sub_request = self.sub_request(request, config)

        return (
            StackJob(
                top=self.top.all_data(sub_request, int(limit / 2) if limit else None),
                bottom=self.bottom.all_data(
                    sub_request, int(limit / 2) if limit else None
                ),
                source_column=self.source_column_config,
            )
            .with_request([request])
            .derive_features([request])
        )

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls, facts: RetrievalJob, requests: list[tuple[StackSource, RetrievalRequest]]
    ) -> RetrievalJob:
        sources = {source.job_group_key() for source, _ in requests}
        if len(sources) != 1:
            raise ValueError(f"Only able to load one {requests} at a time")

        source = requests[0][0]
        if not isinstance(source, cls):
            raise ValueError(f"Only {cls} is supported, received: {source}")

        return source.features_for(facts, requests[0][1])

    def features_for(
        self, facts: RetrievalJob, request: RetrievalRequest
    ) -> RetrievalJob:
        from aligned.local.job import FileFactualJob
        from aligned.retrieval_job import StackJob

        config = self.source_column_config
        sub_request = request

        if config:
            sub_request = self.sub_request(request, config)

        top = self.top.features_for(facts, sub_request).drop_invalid()
        bottom = self.bottom.features_for(facts, sub_request).drop_invalid()

        stack_job = StackJob(top=top, bottom=bottom, source_column=config)

        return FileFactualJob(stack_job, [request], facts)

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        from aligned.retrieval_job import StackJob

        top = self.top.all_between_dates(request, start_date, end_date)
        bottom = self.bottom.all_between_dates(request, start_date, end_date)

        return StackJob(
            top=top,
            bottom=bottom,
            source_column=self.source_column_config,
        )

    def depends_on(self) -> set[FeatureLocation]:
        return self.top.depends_on().union(self.bottom.depends_on())


def request_without_event_timestamp(request: RetrievalRequest) -> RetrievalRequest:
    return RetrievalRequest(
        request.name,
        location=request.location,
        features=request.features,
        entities=request.entities,
        derived_features=request.derived_features,
        aggregated_features=request.aggregated_features,
        features_to_include=request.features_to_include,
        event_timestamp_request=None,
    )


@dataclass
class LoadedAtSource(CodableBatchDataSource):
    source: CodableBatchDataSource
    type_name: str = "loaded_at"

    @property
    def as_markdown(self) -> str:
        source_markdown = self.source.as_markdown
        return f"""### Loaded At Source

Adding a loaded at timestamp to the source:
{source_markdown}
"""  # noqa

    def job_group_key(self) -> str:
        return self.source.job_group_key()

    async def schema(self) -> dict[str, FeatureType]:
        return await self.source.schema()

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        from aligned.retrieval_job import LoadedAtJob

        return LoadedAtJob(
            self.source.all_data(request_without_event_timestamp(request), limit),
            request,
        )

    def all_between_dates(
        self, request: RetrievalRequest, start_date: datetime, end_date: datetime
    ) -> RetrievalJob:
        from aligned.retrieval_job import LoadedAtJob

        return LoadedAtJob(
            self.all_data(request, limit=None),
            request,
        )

    def depends_on(self) -> set[FeatureLocation]:
        return self.source.depends_on()

    async def freshness(self, feature: Feature) -> datetime | None:
        return None

    @classmethod
    def multi_source_features_for(  # type: ignore
        cls: type[CodableBatchDataSource],
        facts: RetrievalJob,
        requests: list[tuple[CodableBatchDataSource, RetrievalRequest]],
    ) -> RetrievalJob:
        return type(requests[0][0]).multi_source_features_for(facts, requests)


@dataclass
class JoinDataSource(CodableBatchDataSource):
    source: CodableBatchDataSource
    left_request: RetrievalRequest
    right_source: CodableBatchDataSource
    right_request: RetrievalRequest
    left_on: list[str]
    right_on: list[str]
    method: Literal["left", "inner", "outer"]

    type_name: str = "join"

    async def schema(self) -> dict[str, FeatureType]:
        left_schema = await self.source.schema()
        right_schema = await self.right_source.schema()

        return {**left_schema, **right_schema}

    def job_group_key(self) -> str:
        return f"join/{self.source.job_group_key()}"

    def all_with_limit(self, limit: int | None) -> RetrievalJob:
        right_job = self.right_source.all_data(
            self.right_request, limit=None
        ).derive_features([self.right_request])

        return (
            self.source.all_data(self.left_request, limit=limit)
            .derive_features([self.left_request])
            .join(
                right_job,
                method=self.method,
                left_on=self.left_on,
                right_on=self.right_on,
            )
            .fill_missing_columns()
        )

    def all_data(self, request: RetrievalRequest, limit: int | None) -> RetrievalJob:
        right_job = self.right_source.all_data(
            self.right_request, limit=None
        ).derive_features([self.right_request])

        return (
            self.source.all_data(self.left_request, limit=limit)
            .derive_features([self.left_request])
            .join(
                right_job,
                method=self.method,
                left_on=self.left_on,
                right_on=self.right_on,
            )
            .fill_missing_columns()
            .aggregate(request)
            .derive_features([request])
        )

    def all_between_dates(
        self,
        request: RetrievalRequest,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        right_et = request.event_timestamp
        assert right_et

        right_job = self.right_source.all_data(
            self.right_request, limit=None
        ).derive_features([self.right_request])

        left_et = self.left_request.event_timestamp
        assert left_et
        left_timestamp = left_et.name

        return (
            self.source.all_between_dates(self.left_request, start_date, end_date)
            .derive_features([self.left_request])
            .join_asof(
                right_job,
                left_event_timestamp=left_timestamp,
                right_event_timestamp=right_et.name,
                left_on=self.left_on,
                right_on=self.right_on,
            )
            .fill_missing_columns()
            .aggregate(request)
            .derive_features([request])
        )

    async def freshness(self, feature: Feature) -> datetime | None:
        left_freshness = await self.source.freshness(feature)
        right_frehsness = await self.right_source.freshness(feature)

        if left_freshness is None:
            return None

        if right_frehsness is None:
            return None

        return min(left_freshness, right_frehsness)

    def join(
        self,
        view: Any,
        on: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        on_left: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        on_right: str | FeatureFactory | list[str] | list[FeatureFactory] | None = None,
        how: Literal["left", "inner", "outer"] = "inner",
    ) -> JoinDataSource:
        if on:
            on_left = on
            on_right = on

        return join_source(self, view, on_left, on_right, how)

    def depends_on(self) -> set[FeatureLocation]:
        return self.source.depends_on().union(self.right_source.depends_on())


class ColumnFeatureMappable:
    mapping_keys: dict[str, str]

    def with_renames(self: T, mapping_keys: dict[str, str]) -> T:
        new = copy(self)
        new.mapping_keys = mapping_keys  # type: ignore
        return new

    def columns_for(self, features: list[Feature]) -> list[str]:
        return [
            self.mapping_keys.get(feature.name, feature.name) for feature in features
        ]

    def feature_identifier_for(self, columns: Iterable[str]) -> list[str]:
        reverse_map = {v: k for k, v in self.mapping_keys.items()}
        return [reverse_map.get(column, column) for column in columns]
