from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from mashumaro.types import SerializableType
from pydantic import BaseModel, ValidationError

from aligned.schemas.codable import Codable

logger = logging.getLogger(__name__)


class SupportedEmbeddingModels:

    types: dict[str, type[EmbeddingModel]]

    _shared: SupportedEmbeddingModels | None = None

    def __init__(self) -> None:
        self.types = {}

        for tran_type in [GensimModel, OpenAiEmbeddingModel, HuggingFaceTransformer]:
            self.add(tran_type)

    def add(self, transformation: type[EmbeddingModel]) -> None:
        self.types[transformation.name] = transformation

    @classmethod
    def shared(cls) -> SupportedEmbeddingModels:
        if cls._shared:
            return cls._shared
        cls._shared = SupportedEmbeddingModels()
        return cls._shared


class EmbeddingModel(Codable, SerializableType):
    name: str

    @property
    def embedding_size(self) -> int | None:
        return None

    def _serialize(self) -> dict:
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> EmbeddingModel:
        name_type = value['name']
        del value['name']
        data_class = SupportedEmbeddingModels.shared().types[name_type]
        with suppress(AttributeError):
            if data_class.dtype:
                del value['dtype']

        return data_class.from_dict(value)

    async def load_model(self):
        pass

    async def vectorise_pandas(self, texts: pd.Series) -> pd.Series:
        raise NotImplementedError(type(self))

    async def vectorise_polars(self, texts: pl.LazyFrame, text_key: str, output_key: str) -> pl.Expr:
        raise NotImplementedError(type(self))

    @staticmethod
    def gensim(model_name: str, config: GensimConfig | None = None) -> GensimModel:
        return GensimModel(model_name=model_name, config=config or GensimConfig())

    @staticmethod
    def openai(
        model_name: str = 'text-embedding-ada-002', api_token_env_key: str = 'OPENAI_API_KEY'
    ) -> OpenAiEmbeddingModel:
        return OpenAiEmbeddingModel(model=model_name, api_token_env_key=api_token_env_key)

    @staticmethod
    def huggingface(model_name: str) -> HuggingFaceTransformer:
        return HuggingFaceTransformer(model=model_name)


@dataclass
class GensimConfig(Codable):
    to_lowercase: bool = field(default=False)
    deaccent: bool = field(default=False)
    encoding: str = field(default='utf8')
    errors: str = field(default='strict')


@dataclass
class GensimModel(EmbeddingModel):

    model_name: str
    config: GensimConfig = field(default_factory=GensimConfig)

    loaded_model: Any = field(default=None)
    name: str = 'gensim'

    async def vectorise_pandas(self, texts: pd.Series) -> pd.Series:
        if not self.loaded_model:
            await self.load_model()

        from gensim.utils import tokenize

        def token(text: str) -> list[str]:
            return list(
                tokenize(
                    text,
                    lowercase=self.config.to_lowercase,
                    deacc=self.config.deaccent,
                    encoding=self.config.encoding,
                    errors=self.config.errors,
                )
            )

        tokens = texts.apply(token)

        def vector(tokens: list[str]):
            vector = np.zeros(self.loaded_model.vector_size)
            n = 0
            for token in tokens:
                if token in self.loaded_model:
                    vector += self.loaded_model[token]
                    n += 1
            if n > 0:
                vector = vector / n

            return vector

        return tokens.apply(vector)

    async def vectorise_polars(self, texts: pl.LazyFrame, text_key: str, output_key: str) -> pl.LazyFrame:
        if not self.loaded_model:
            await self.load_model()

        from gensim.utils import tokenize

        def token(text: str) -> list[str]:
            return list(
                tokenize(
                    text,
                    lowercase=self.config.to_lowercase,
                    deacc=self.config.deaccent,
                    encoding=self.config.encoding,
                    errors=self.config.errors,
                )
            )

        tokenised_text = texts.with_columns([pl.col(text_key).apply(token).alias(f'{text_key}_tokens')])

        def vector(tokens: list[str]) -> list[float]:
            logger.info('Computing vector', tokens)
            vector = np.zeros(self.loaded_model.vector_size)
            n = 0
            for token in tokens:
                if token in self.loaded_model:
                    vector += self.loaded_model[token]
                    n += 1
            if n > 0:
                vector = vector / n

            return vector.tolist()

        return tokenised_text.with_columns(
            [pl.col(f'{text_key}_tokens').apply(vector, return_dtype=pl.List(pl.Float64)).alias(output_key)]
        )

    async def load_model(self):
        import gensim.downloader as gensim_downloader

        self.loaded_model = gensim_downloader.load(self.model_name)
        logger.info(f'Loaded model {self.model_name}')


class OpenAiEmbedding(BaseModel):
    index: int
    embedding: list[float]


class OpenAiResponse(BaseModel):
    data: list[OpenAiEmbedding]


@dataclass
class OpenAiEmbeddingModel(EmbeddingModel):

    api_token_env_key: str = field(default='OPENAI_API_KEY')
    model: str = field(default='text-embedding-ada-002')
    name: str = 'openai'

    @property
    def embedding_size(self) -> int | None:
        return 768

    async def embeddings(self, input: list[str]) -> OpenAiResponse:
        # import openai
        import os

        # openai.api_key = os.environ[self.api_token_env_key]
        # return openai.Embedding.create(
        #     model="text-embedding-ada-002",
        #     input=input
        # )
        from httpx import AsyncClient

        api_token = os.environ[self.api_token_env_key]

        assert isinstance(input, list)
        for item in input:
            assert item is not None
            assert isinstance(item, str)
            assert len(item) > 0
            assert len(item) / 3 < 7000

        try:
            async with AsyncClient() as client:
                json = {'model': self.model, 'input': input}
                response = await client.post(
                    url='https://api.openai.com/v1/embeddings',
                    json=json,
                    headers={'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'},
                )
                response.raise_for_status()
        except Exception as e:
            logger.error(f'Error calling OpenAi Embeddings API: {e}')
            logger.error(f'Input: {input}')
            raise e

        try:
            return OpenAiResponse(**response.json())
        except ValidationError as e:
            logger.error(f'Response: {response.json()}')
            logger.error(f'Error parsing OpenAi Embeddings API response: {e}')
            raise e

    async def load_model(self):
        pass

    async def vectorise_pandas(self, texts: pd.Series) -> pd.Series:
        data = await self.embeddings(texts.tolist())
        return pd.Series([embedding.embedding for embedding in data.data])

    async def vectorise_polars(self, texts: pl.LazyFrame, text_key: str, output_key: str) -> pl.Expr:
        async def embed(text: pl.Series) -> pl.Series:
            data = await self.embeddings(text.to_list())
            return pl.Series([embedding.embedding for embedding in data.data])

        return pl.col(text_key).map_batches(lambda text: embed(text)).alias(output_key)


@dataclass
class HuggingFaceTransformer(EmbeddingModel):

    model: str
    name: str = 'huggingface'
    loaded_model: Any = field(default=None)

    @property
    def embedding_size(self) -> int | None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model)

        return len(model.encode(['test'])[0])

    async def load_model(self):
        from sentence_transformers import SentenceTransformer

        self.loaded_model = SentenceTransformer(self.model)

    async def vectorise_polars(self, texts: pl.LazyFrame, text_key: str, output_key: str) -> pl.Expr:
        if self.loaded_model is None:
            await self.load_model()

        return (
            pl.col(text_key)
            .map_batches(lambda text: pl.Series(self.loaded_model.encode(text.to_list())))
            .alias(output_key)
        )

    async def vectorise_pandas(self, texts: pd.Series) -> pd.Series:
        if self.loaded_model is None:
            await self.load_model()
        return pd.Series(self.loaded_model.encode(texts.tolist()).tolist())
