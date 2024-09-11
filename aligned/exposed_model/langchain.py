from dataclasses import dataclass, field
from typing import Any, AsyncIterable
from aligned.compiler.model import ModelContractWrapper
from aligned.data_source.batch_data_source import CodableBatchDataSource, CustomMethodDataSource
from aligned.feature_view.feature_view import FeatureViewWrapper
import polars as pl
from datetime import datetime

from aligned.exposed_model.interface import ExposedModel, PromptModel, StreamablePredictor, RetrivalJob
from aligned.feature_store import ContractStore, ModelFeatureStore
from aligned.request.retrival_request import RetrivalRequest
from aligned.schemas.feature import FeatureLocation

try:
    from langchain_core.language_models.base import LanguageModelLike
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
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

    def __str__(self) -> str:
        return f"Aligned Retriver - Loading {self.number_of_docs} from '{self.index_name}'"

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

        assert (
            embed_model.has_exposed_model()
        ), f"The model {index.model.name} do not have an exposed model. Which means we can not use it."

        exposed_model = embed_model.model.exposed_model

        if isinstance(exposed_model, PromptModel) and exposed_model.precomputed_prompt_key:
            input_name = exposed_model.precomputed_prompt_key
        else:
            inputs = list(embed_model.model.feature_references())
            assert len(inputs) == 1, (
                f"Model have more than one inputs: {len(inputs)}. "
                f"Unclear what to name the query: '{query}'. "
                'This can be fixed by making sure the underlying model is a '
                '`PromptModel` with a `precomputed_prompt_key`.'
            )
            input_name = inputs[0].name

        embedding = await store.model(embed_model.model.name).predict_over({input_name: [query]}).to_polars()

        embedding_output = [
            feature.name
            for feature in embed_model.prediction_request().all_returned_features
            if not feature.dtype.is_embedding or feature.dtype.is_array
        ]

        documents = await index.nearest_n_to(
            entities=embedding.select(pl.exclude(input_name)), number_of_records=self.number_of_docs
        ).to_polars()

        documents = documents.with_columns(
            page_content=pl.concat_str(
                [pl.col(col).cast(pl.String) for col in embedding_output], separator='\n\n'
            )
        )

        return [Document(**doc) for doc in documents.to_dicts()]


@dataclass
class LangChain(ExposedModel, StreamablePredictor):

    chain_bytes: bytes
    chain_output: str
    output_key: str

    depends_on_data: list[FeatureLocation] = field(default_factory=list)

    @property
    def chain(self) -> LanguageModelLike:
        from dill import loads

        return loads(self.chain_bytes)

    @property
    def as_markdown(self) -> str:
        return f"A LangChain model looking like {str(self.chain)}"

    @staticmethod
    def from_chain(
        chain: LanguageModelLike,
        chain_output: str = 'answer',
        output_key: str = 'answer',
        depends_on: list[FeatureLocation | FeatureViewWrapper | ModelContractWrapper] | None = None,
    ) -> 'LangChain':
        from dill import dumps

        return LangChain(
            dumps(chain),
            output_key=output_key,
            chain_output=chain_output,
            depends_on_data=[]
            if depends_on is None
            else [loc if isinstance(loc, FeatureLocation) else loc.location for loc in depends_on],
        )

    @property
    def version(self) -> str:
        from hashlib import sha256

        return sha256(self.chain_bytes, usedforsecurity=False).hexdigest()

    async def depends_on(self) -> list[FeatureLocation]:
        return self.depends_on_data

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        responses = []
        pred_view = store.model.predictions_view
        df = await values.to_polars()
        for question in df.to_dicts():
            responses.append((await self.chain.ainvoke(question))[self.chain_output])

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

            if self.output_key != self.chain_output and self.chain_output in value:
                value[self.output_key] = value[self.chain_output]
            yield value


def web_chunks_source(pages: list[str]) -> CodableBatchDataSource:
    async def load(request: RetrivalRequest) -> pl.LazyFrame:
        import polars as pl
        from datetime import timezone, datetime
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

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
            loaded_at=pl.lit(datetime.now(tz=timezone.utc)), chunk_hash=pl.col('page_content').hash()
        ).lazy()

    return CustomMethodDataSource.from_load(load)


def file_chunks_source(directory: str, glob: str = '**') -> CodableBatchDataSource:
    async def load(request: RetrivalRequest) -> pl.LazyFrame:
        import polars as pl
        from datetime import timezone, datetime
        from langchain_community.document_loaders import DirectoryLoader, TextLoader, PythonLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
        import logging

        logger = logging.getLogger(__name__)

        loader_cls = TextLoader
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

        if glob.endswith('.py'):
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
            loaded_at=pl.lit(datetime.now(tz=timezone.utc)), chunk_hash=pl.col('page_content').hash()
        ).lazy()

    return CustomMethodDataSource.from_load(load)
