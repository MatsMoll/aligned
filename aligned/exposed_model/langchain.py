from dataclasses import dataclass
from typing import Any, AsyncIterable
from aligned.data_source.batch_data_source import BatchDataSource, CustomMethodDataSource
from aligned.feature_view.feature_view import FeatureViewWrapper
import polars as pl
from datetime import datetime

from aligned.exposed_model.interface import ExposedModel, StreamablePredictor, RetrivalJob
from aligned.feature_store import ContractStore, ModelFeatureStore

try:
    from langchain_core.language_models.base import LanguageModelLike
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_text_splitters import TextSplitter
    from langchain_core.documents.base import Document
except ModuleNotFoundError:

    class BaseRetriever:
        pass

    class LanguageModelLike:
        pass


class AlignedRetriver(BaseRetriever):

    store: ContractStore
    index_name: str
    number_of_docs: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: 'CallbackManagerForRetrieverRun'
    ) -> list['Document']:
        raise NotImplementedError()

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: 'AsyncCallbackManagerForRetrieverRun'
    ) -> list['Document']:

        store = self.store
        index = store.vector_index(self.index_name)
        embed_model = store.model(index.model.name)

        input = list(embed_model.model.feature_references())[0]

        embedding = await store.model(embed_model.model.name).predict_over({input.name: [query]}).to_polars()

        documents = await index.nearest_n_to(
            entities=embedding.select(pl.exclude(input.name)), number_of_records=self.number_of_docs
        ).to_polars()

        return [Document(**doc) for doc in documents.to_dicts()]


@dataclass
class LangChain(ExposedModel, StreamablePredictor):

    chain_bytes: bytes
    output_key: str

    @property
    def chain(self) -> LanguageModelLike:
        from dill import loads

        return loads(self.chain_bytes)

    @staticmethod
    def from_chain(chain: LanguageModelLike, output_key: str = 'answer') -> 'LangChain':
        from dill import dumps

        return LangChain(dumps(chain), output_key=output_key)

    @property
    def version(self) -> str:
        from hashlib import sha256

        return sha256(self.chain_bytes, usedforsecurity=False).hexdigest()

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        responses = []
        pred_view = store.model.predictions_view

        df = await values.to_polars()
        for question in df.to_dicts():
            responses.append((await self.chain.ainvoke(question))[self.output_key])

        if pred_view.model_version_column:
            df = df.with_columns(pl.lit(self.version).alias(pred_view.model_version_column.name))

        if pred_view.event_timestamp:
            df = df.with_columns(pl.lit(datetime.utcnow()).alias(pred_view.event_timestamp.name))

        return df.hstack([pl.Series(name=self.output_key, values=responses)])

    async def stream_predict(self, input: dict[str, Any]) -> AsyncIterable[dict[str, Any]]:
        async for output in self.chain.astream(input):
            try:
                if isinstance(output, dict):
                    value = output
                else:
                    value = output.model_dump()
            except AttributeError:
                value = output.dict()
            yield value


def web_chunks_source(pages: list[str], splitter: TextSplitter | None = None) -> BatchDataSource:
    async def load(request) -> pl.LazyFrame:
        import polars as pl
        from datetime import timezone, datetime
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        all_splits = []

        splitter = splitter or RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        for page in pages:
            loader = WebBaseLoader(page)
            data = loader.load()

            all_splits.extend(splitter.split_documents(data))

        flattend_data = []
        for doc in all_splits:
            flattend_data.append(dict(page_content=doc.page_content, **doc.metadata))

        df = pl.DataFrame(flattend_data)
        return df.with_columns(
            loaded_at=pl.lit(datetime.now(tz=timezone.utc)), chunk_hash=pl.col('page_content').hash()
        )

    return CustomMethodDataSource.from_load(load)
