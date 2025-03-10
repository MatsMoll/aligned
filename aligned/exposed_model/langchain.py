from dataclasses import dataclass, field
from typing import Any, AsyncIterable
from aligned.compiler.model import ModelContractWrapper
from aligned.data_source.batch_data_source import (
    CodableBatchDataSource,
    CustomMethodDataSource,
)
from aligned.feature_view.feature_view import FeatureViewWrapper
import polars as pl
from datetime import datetime

from aligned.exposed_model.interface import (
    ExposedModel,
    StreamablePredictor,
    RetrievalJob,
)
from aligned.feature_store import ModelFeatureStore
from aligned.request.retrieval_request import RetrievalRequest
from aligned.schemas.feature import FeatureLocation
from aligned.lazy_imports import langchain_core


@dataclass
class LangChain(ExposedModel, StreamablePredictor):
    chain_bytes: bytes
    chain_output: str
    output_key: str

    depends_on_data: list[FeatureLocation] = field(default_factory=list)
    model_type: str = "langchain"

    @property
    def chain(self) -> "langchain_core.language_models.base.LanguageModelLike":
        from dill import loads

        return loads(self.chain_bytes)

    @property
    def as_markdown(self) -> str:
        return f"A LangChain model looking like {str(self.chain)}"

    @staticmethod
    def from_chain(
        chain: "langchain_core.language_models.base.LanguageModelLike",
        chain_output: str = "answer",
        output_key: str = "answer",
        depends_on: list[FeatureLocation | FeatureViewWrapper | ModelContractWrapper]
        | None = None,
    ) -> "LangChain":
        from dill import dumps

        return LangChain(
            dumps(chain),
            output_key=output_key,
            chain_output=chain_output,
            depends_on_data=[]
            if depends_on is None
            else [
                loc if isinstance(loc, FeatureLocation) else loc.location
                for loc in depends_on
            ],
        )

    @property
    def version(self) -> str:
        from hashlib import sha256

        return sha256(self.chain_bytes, usedforsecurity=False).hexdigest()

    async def depends_on(self) -> list[FeatureLocation]:
        return self.depends_on_data

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        responses = []
        pred_view = store.model.predictions_view
        df = await values.to_polars()
        for question in df.to_dicts():
            responses.append((await self.chain.ainvoke(question))[self.chain_output])

        if pred_view.model_version_column:
            df = df.with_columns(
                pl.lit(self.version).alias(pred_view.model_version_column.name)
            )

        if pred_view.event_timestamp:
            df = df.with_columns(
                pl.lit(datetime.utcnow()).alias(pred_view.event_timestamp.name)
            )

        return df.hstack([pl.Series(name=self.output_key, values=responses)])

    async def stream_predict(
        self, input: dict[str, Any]
    ) -> AsyncIterable[dict[str, Any]]:
        async for output in self.chain.astream(input):
            try:
                if isinstance(output, dict):
                    value = output
                else:
                    value = output.model_dump()
            except AttributeError:
                value = output.dict()  # type: ignore

            if self.output_key != self.chain_output and self.chain_output in value:
                value[self.output_key] = value[self.chain_output]
            yield value


def web_chunks_source(pages: list[str]) -> CodableBatchDataSource:
    async def load(request: RetrievalRequest) -> pl.LazyFrame:
        import polars as pl
        from datetime import timezone, datetime
        from langchain_community.document_loaders import WebBaseLoader  # type: ignore
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        all_splits = []

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        for page in pages:
            loader = WebBaseLoader(page)
            data = loader.load()

            all_splits.extend(splitter.split_documents(data))

        flattend_data = []
        for doc in all_splits:
            flattend_data.append(dict(page_content=doc.page_content, **doc.metadata))

        df = pl.DataFrame(flattend_data)
        return df.with_columns(
            loaded_at=pl.lit(datetime.now(tz=timezone.utc)),
            chunk_hash=pl.col("page_content").hash(),
        ).lazy()

    return CustomMethodDataSource.from_load(load)


def file_chunks_source(directory: str, glob: str = "**") -> CodableBatchDataSource:
    async def load(request: RetrievalRequest) -> pl.LazyFrame:
        import polars as pl
        from datetime import timezone, datetime
        from langchain_community.document_loaders import (  # type: ignore
            DirectoryLoader,
            TextLoader,
            PythonLoader,
        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter, Language  # type: ignore
        import logging

        logger = logging.getLogger(__name__)

        loader_cls = TextLoader
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

        if glob.endswith(".py"):
            loader_cls = PythonLoader
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON, chunk_size=500, chunk_overlap=0
            )

        logger.info(loader_cls)
        logger.info(splitter)

        loader = DirectoryLoader(directory, glob=glob, loader_cls=loader_cls)

        flattend_data = []
        for doc in splitter.split_documents(loader.load()):
            flattend_data.append(dict(page_content=doc.page_content, **doc.metadata))

        df = pl.DataFrame(flattend_data)
        return df.with_columns(
            loaded_at=pl.lit(datetime.now(tz=timezone.utc)),
            chunk_hash=pl.col("page_content").hash(),
        ).lazy()

    return CustomMethodDataSource.from_load(load)
