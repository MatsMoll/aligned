from __future__ import annotations
from typing import Callable, Literal

from dataclasses import dataclass, field

import polars as pl
from aligned.lazy_imports import pandas as pd
from aligned.schemas.codable import Codable


def camel_to_snake_case(column: str) -> str:
    return "".join(
        ["_" + char.lower() if char.isupper() else char for char in column]
    ).lstrip("_")


def snake_to_camel(column: str) -> str:
    pascal = snake_to_pascal(column)
    return pascal[0].lower() + pascal[1:]


def snake_to_pascal(column: str) -> str:
    return "".join(
        [s[0].upper() + s[1:].lower() if s else s for s in column.split("_")]
    )


@dataclass
class Renamer(Codable):
    renamer_type: Literal[
        "camel_to_snake", "snake_to_pascal", "snake_to_camel", "noop"
    ] = field(default="noop")
    mapping: dict[str, str] | None = field(default=None)

    def inverse(self) -> Renamer:
        renamer_type_map = {
            "camel_to_snake": "snake_to_camel",
            "snake_to_pascal": "camel_to_snake",
            "snake_to_camel": "camel_to_snake",
            "noop": "noop",
        }
        return Renamer(
            renamer_type_map[self.renamer_type],  # type: ignore
            mapping={value: key for key, value in self.mapping.items()}
            if self.mapping
            else None,
        )

    def rename_func(self) -> Callable[[str], str]:
        return {
            "camel_to_snake": camel_to_snake_case,
            "snake_to_pascal": snake_to_pascal,
            "snake_to_camel": snake_to_camel,
        }.get(self.renamer_type, lambda col: col)

    def rename(self, col: str) -> str:
        if self.renamer_type == "noop":
            return (self.mapping or {}).get(col, col)

        if self.mapping and col in self.mapping:
            return self.mapping[col]
        else:
            return self.rename_func()(col)

    def rename_polars(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if self.renamer_type == "noop" and not self.mapping:
            return df

        if self.mapping:
            df = df.rename(self.mapping)

        return df.rename(self.rename_func())

    def rename_pandas(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.renamer_type == "noop" and not self.mapping:
            return df

        if self.mapping:
            df = df.rename(columns=self.mapping)
        return df.rename(columns=self.rename_func())

    @staticmethod
    def noop(mapping: dict[str, str] | None = None) -> Renamer:
        return Renamer(renamer_type="noop", mapping=mapping)

    @staticmethod
    def camel_to_snake(mapping: dict[str, str] | None = None) -> Renamer:
        return Renamer(renamer_type="camel_to_snake", mapping=mapping)

    @staticmethod
    def pascal_to_snake(mapping: dict[str, str] | None = None) -> Renamer:
        return Renamer(renamer_type="camel_to_snake", mapping=mapping)

    @staticmethod
    def snake_to_pascal(mapping: dict[str, str] | None = None) -> Renamer:
        return Renamer(renamer_type="snake_to_pascal", mapping=mapping)

    @staticmethod
    def snake_to_camel(mapping: dict[str, str] | None = None) -> Renamer:
        return Renamer(renamer_type="snake_to_camel", mapping=mapping)
