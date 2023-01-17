from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from mashumaro.types import SerializableType

from aligned.schemas.codable import Codable


class SupportedTextModels:

    types: dict[str, type[TextVectoriserModel]]

    _shared: SupportedTextModels | None = None

    def __init__(self) -> None:
        self.types = {}

        for tran_type in [GensimModel]:
            self.add(tran_type)

    def add(self, transformation: type[TextVectoriserModel]) -> None:
        self.types[transformation.name] = transformation

    @classmethod
    def shared(cls) -> SupportedTextModels:
        if cls._shared:
            return cls._shared
        cls._shared = SupportedTextModels()
        return cls._shared


class TextVectoriserModel(Codable, SerializableType):
    name: str

    def _serialize(self) -> dict:
        return self.to_dict()

    @classmethod
    def _deserialize(cls, value: dict) -> TextVectoriserModel:
        name_type = value['name']
        del value['name']
        data_class = SupportedTextModels.shared().types[name_type]
        with suppress(AttributeError):
            if data_class.dtype:
                del value['dtype']

        return data_class.from_dict(value)

    async def load_model(self):
        pass

    async def vectorise_pandas(self, texts: pd.Series) -> pd.Series:
        pass

    async def vectorise_polars(self, texts: pl.LazyFrame, text_key: str, output_key: str) -> pl.LazyFrame:
        pass

    @staticmethod
    def gensim(model_name: str, config: GensimConfig | None = None) -> GensimModel:
        return GensimModel(model_name=model_name, config=config or GensimConfig())


@dataclass
class GensimConfig(Codable):
    to_lowercase: bool = field(default=False)
    deaccent: bool = field(default=False)
    encoding: str = field(default='utf8')
    errors: str = field(default='strict')


@dataclass
class GensimModel(TextVectoriserModel):

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
