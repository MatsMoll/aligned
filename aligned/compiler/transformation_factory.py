from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta  # noqa: TC003
from typing import TYPE_CHECKING, Any, Callable

import polars as pl

from aligned import AwsS3Config
from aligned.lazy_imports import pandas as pd
from aligned.compiler.feature_factory import (
    FeatureFactory,
    Transformation,
    TransformationFactory,
)
from aligned.schemas.feature import FeatureReference, FeatureType
from aligned.schemas.transformation import (
    FillNaValuesColumns,
    LiteralValue,
    EmbeddingModel,
    BinaryOperators,
    Expression,
    UnaryFunction,
)

if TYPE_CHECKING:
    from aligned.feature_store import ContractStore

logger = logging.getLogger(__name__)


@dataclass
class OrdinalFactory(TransformationFactory):
    orders: list[str]
    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Ordinal

        return Ordinal(self.feature.name, self.orders)


@dataclass
class ArrayAtIndexFactory(TransformationFactory):
    feature: FeatureFactory
    index: int

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import ArrayAtIndex

        return ArrayAtIndex(self.feature.name, self.index)


@dataclass
class ArrayContainsAnyFactory(TransformationFactory):
    values: LiteralValue
    in_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.in_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import ArrayContainsAny

        return ArrayContainsAny(self.in_feature.name, self.values)


@dataclass
class BinaryFactory(TransformationFactory):
    left: Any
    right: Any
    operation: BinaryOperators

    @property
    def using_features(self) -> list[FeatureFactory]:
        features = []
        if isinstance(self.left, FeatureFactory):
            features.append(self.left)
            features.extend(
                feat
                for feat in self.left.feature_dependencies()
                if feat._name is not None
            )
        if isinstance(self.right, FeatureFactory):
            features.append(self.right)
            features.extend(
                feat
                for feat in self.right.feature_dependencies()
                if feat._name is not None
            )
        return features

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import BinaryTransformation, Expression

        if isinstance(self.left, FeatureFactory):
            left = self.left.to_expression()
        else:
            left = Expression(literal=LiteralValue.from_value(self.left))

        if isinstance(self.right, FeatureFactory):
            right = self.right.to_expression()
        else:
            right = Expression(literal=LiteralValue.from_value(self.right))

        return BinaryTransformation(left, right, operator=self.operation)


@dataclass
class Split(TransformationFactory):
    pattern: str
    from_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.from_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Split as SplitTransformation

        return SplitTransformation(
            Expression.from_value(self.from_feature), self.pattern
        )


@dataclass
class OllamaEmbedding(TransformationFactory):
    model: str
    from_feature: FeatureFactory
    host_env: str | None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.from_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import (
            OllamaEmbedding as OllamaEmbeddingTransformation,
        )

        return OllamaEmbeddingTransformation(
            self.from_feature.name, self.model, self.host_env
        )


@dataclass
class OllamaGenerate(TransformationFactory):
    model: str
    system: str | None
    prompt_feature: FeatureFactory
    host_env: str | None

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.prompt_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import (
            OllamaGenerate as OllamaGenerateTransformation,
        )

        return OllamaGenerateTransformation(
            self.prompt_feature.name, self.model, self.system or "", self.host_env
        )


@dataclass
class DateComponentFactory(TransformationFactory):
    component: str
    feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import DateComponent as DCTransformation

        return DCTransformation(self.feature.name, self.component)


@dataclass
class TimeDifferanceFactory(TransformationFactory):
    first_feature: FeatureFactory
    second_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.first_feature, self.second_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import TimeDifference as TDTransformation

        return TDTransformation(self.first_feature.name, self.second_feature.name)


@dataclass
class ReplaceFactory(TransformationFactory):
    values: dict[str, str]
    source_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.source_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import ReplaceStrings

        values = list(self.values.items())
        return ReplaceStrings(self.source_feature.name, values)


@dataclass
class ToNumericalFactory(TransformationFactory):
    from_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.from_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import (
            ToNumerical as ToNumericalTransformation,
        )

        return ToNumericalTransformation(self.from_feature.name)


@dataclass
class IsInFactory(TransformationFactory):
    feature: FeatureFactory
    values: list

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import IsIn as IsInTransformation

        return IsInTransformation(self.values, self.feature.to_expression())


class FillNaStrategy:
    def compile(self) -> Any:
        pass


@dataclass
class ConstantFillNaStrategy(FillNaStrategy):
    value: Any

    def compile(self) -> Any:
        return self.value


@dataclass
class FillMissingFactory(TransformationFactory):
    feature: FeatureFactory
    value: LiteralValue | FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        if isinstance(self.value, LiteralValue):
            return [self.feature]
        else:
            return [self.feature, self.value]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import FillNaValues

        if isinstance(self.value, LiteralValue):
            return FillNaValues(
                key=self.feature.name, value=self.value, dtype=self.feature.dtype
            )
        else:
            return FillNaValuesColumns(
                key=self.feature.name,
                fill_key=self.value.name,
                dtype=self.feature.dtype,
            )


@dataclass
class UnaryFactory(TransformationFactory):
    inner: FeatureFactory
    func: UnaryFunction

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.inner]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import UnaryTransformation

        return UnaryTransformation(
            inner=Expression.from_value(self.inner), func=self.func
        )


@dataclass
class MapRowTransformation(TransformationFactory):
    dtype: FeatureFactory
    method: (
        Callable[[Any, dict, ContractStore], Any] | Callable[[dict, ContractStore], Any]
    )
    _using_features: list[FeatureFactory]

    @property
    def using_features(self) -> list[FeatureFactory]:
        return self._using_features

    def compile(self) -> Transformation:
        import inspect
        import types
        import dill
        from aligned.schemas.transformation import PolarsMapRowTransformation

        if (
            isinstance(self.method, types.LambdaType)
            and self.method.__name__ == "<lambda>"
        ):
            raise NotImplementedError(type(self))

        function_name = dill.source.getname(self.method)
        assert isinstance(function_name, str), "Need a function name"
        raw_code = inspect.getsource(self.method)

        code = ""

        indents: int | None = None
        start_signature = f"def {function_name}"
        strip_self = False

        for line in raw_code.splitlines(keepends=True):
            if strip_self:
                code += line.replace("self,", "")
                strip_self = False
            elif start_signature in line:
                stripped = line.lstrip()
                indents = len(line) - len(stripped)
                stripped = stripped.replace("self,", "")
                code += stripped

                if line.endswith("(\n"):
                    strip_self = True
            elif indents:
                if len(line) > indents:
                    code += line[:indents].lstrip() + line[indents:]
                else:
                    code += line

        return PolarsMapRowTransformation(
            code=code,
            function_name=function_name,
            dtype=self.dtype.dtype,
        )


@dataclass
class PandasTransformationFactory(TransformationFactory):
    dtype: FeatureFactory
    method: Callable[[pd.DataFrame, ContractStore], pd.Series]
    _using_features: list[FeatureFactory]

    @property
    def using_features(self) -> list[FeatureFactory]:
        return self._using_features

    def compile(self) -> Transformation:
        import inspect
        import types

        import dill

        from aligned.schemas.transformation import (
            PandasFunctionTransformation,
            PandasLambdaTransformation,
        )

        if (
            isinstance(self.method, types.LambdaType)
            and self.method.__name__ == "<lambda>"
        ):
            return PandasLambdaTransformation(
                method=dill.dumps(self.method),
                code=inspect.getsource(self.method).strip(),
                dtype=self.dtype.dtype,
            )
        else:
            function_name = dill.source.getname(self.method)
            assert isinstance(function_name, str), "Need a function name"
            raw_code = inspect.getsource(self.method)

            code = ""

            indents: int | None = None
            start_signature = f"def {function_name}"

            for line in raw_code.splitlines(keepends=True):
                if indents:
                    if len(line) > indents:
                        code += line[:indents].lstrip() + line[indents:]
                    else:
                        code += line

                if start_signature in line:
                    stripped = line.lstrip()
                    indents = len(line) - len(stripped)
                    stripped = stripped.replace(
                        f"{start_signature}(self,", f"{start_signature}("
                    )
                    code += stripped

            return PandasFunctionTransformation(
                code=code,
                function_name=function_name,
                dtype=self.dtype.dtype,
            )


@dataclass
class PolarsTransformationFactory(TransformationFactory):
    dtype: FeatureFactory
    method: pl.Expr | Callable[[pl.LazyFrame, pl.Expr, ContractStore], pl.LazyFrame]
    _using_features: list[FeatureFactory]

    @property
    def using_features(self) -> list[FeatureFactory]:
        return self._using_features

    def compile(self) -> Transformation:
        import inspect
        import types

        import dill

        from aligned.schemas.transformation import (
            PolarsFunctionTransformation,
            PolarsLambdaTransformation,
            PolarsExpression,
        )

        if isinstance(self.method, pl.Expr):
            return PolarsExpression(
                self.method.meta.serialize(format="json"), dtype=self.dtype.dtype
            )
        else:
            function_name = dill.source.getname(self.method)
            assert isinstance(function_name, str), "Need a function name"
            raw_code = inspect.getsource(self.method)

            code = ""

            indents: int | None = None
            strip_self = False
            start_signature = f"def {function_name}"

            for line in raw_code.splitlines(keepends=True):
                if strip_self:
                    code += line.replace("self,", "")
                    strip_self = False
                elif start_signature in line:
                    stripped = line.lstrip()
                    indents = len(line) - len(stripped)
                    stripped = stripped.replace("self,", "")
                    code += stripped

                    if line.endswith("(\n"):
                        strip_self = True
                elif indents:
                    if len(line) > indents:
                        code += line[:indents].lstrip() + line[indents:]
                    else:
                        code += line

        if (
            isinstance(self.method, types.LambdaType)
            and self.method.__name__ == "<lambda>"
        ):
            return PolarsLambdaTransformation(
                method=dill.dumps(self.method),
                code=code.strip(),
                dtype=self.dtype.dtype,
            )
        else:
            function_name = dill.source.getname(self.method)
            assert isinstance(
                function_name, str
            ), f"Expected string got {type(function_name)}"
            return PolarsFunctionTransformation(
                code=code,
                function_name=function_name,
                dtype=self.dtype.dtype,
            )


class AggregatableTransformation:
    def copy(self) -> "AggregatableTransformation":
        raise NotImplementedError(type(self))


@dataclass
class MeanTransfomrationFactory(TransformationFactory, AggregatableTransformation):
    feature: FeatureFactory
    over: timedelta | None = field(default=None)
    group_by: list[FeatureFactory] | None = field(default=None)

    @property
    def using_features(self) -> list[FeatureFactory]:
        if self.group_by:
            return [self.feature] + self.group_by
        else:
            return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import MeanAggregation

        return MeanAggregation(key=self.feature.name)

    def copy(self) -> "MeanTransfomrationFactory":
        return MeanTransfomrationFactory(self.feature, self.over, self.group_by)


@dataclass
class WordVectoriserFactory(TransformationFactory):
    feature: FeatureFactory
    model: EmbeddingModel

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import WordVectoriser

        return WordVectoriser(self.feature.name, self.model)


@dataclass
class LoadImageBytesFactory(TransformationFactory):
    url_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.url_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import LoadImageUrlBytes

        return LoadImageUrlBytes(self.url_feature.name)


@dataclass
class LoadImageFactory(TransformationFactory):
    url_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.url_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import LoadImageUrl

        return LoadImageUrl(self.url_feature.name)


@dataclass
class GrayscaleImageFactory(TransformationFactory):
    image_feature: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.image_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import GrayscaleImage

        return GrayscaleImage(self.image_feature.name)


@dataclass
class AppendStrings(TransformationFactory):
    first_feature: FeatureFactory
    second_feature: FeatureFactory | LiteralValue
    separator: str = field(default="")

    @property
    def using_features(self) -> list[FeatureFactory]:
        if isinstance(self.second_feature, LiteralValue):
            return [self.first_feature]
        else:
            return [self.first_feature, self.second_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import AppendConstString, AppendStrings

        if isinstance(self.second_feature, LiteralValue):
            return AppendConstString(
                self.first_feature.name, self.second_feature.python_value
            )
        else:
            return AppendStrings(
                self.first_feature.name, self.second_feature.name, self.separator
            )


@dataclass
class StructFieldFactory(TransformationFactory):
    struct_feature: FeatureFactory
    field_name: str

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.struct_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import StructField

        return StructField(self.struct_feature.name, self.field_name)


@dataclass
class JsonPathFactory(TransformationFactory):
    json_feature: FeatureFactory
    path: str

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.json_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import JsonPath

        return JsonPath(self.json_feature.name, self.path)


@dataclass
class PresignedAwsUrlFactory(TransformationFactory):
    aws_config: AwsS3Config
    url_feature: FeatureFactory
    max_age_seconds: int = field(default=30)

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.url_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import PresignedAwsUrl

        return PresignedAwsUrl(
            self.aws_config, self.url_feature.name, self.max_age_seconds
        )


@dataclass
class PrependConstString(TransformationFactory):
    first_feature: str
    second_feature: FeatureFactory
    separator: str = field(default="")

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.second_feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import PrependConstString

        return PrependConstString(self.first_feature, self.second_feature.name)


@dataclass
class ClipFactory(TransformationFactory):
    feature: FeatureFactory
    lower_bound: int | float
    upper_bound: int | float

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.feature]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import Clip

        return Clip(
            Expression.from_value(self.feature),
            LiteralValue.from_value(self.lower_bound),
            LiteralValue.from_value(self.upper_bound),
        )


@dataclass
class LoadFeature(TransformationFactory):
    entities: dict[str, FeatureFactory]
    feature: FeatureReference
    dtype: FeatureType

    @property
    def using_features(self) -> list[FeatureFactory]:
        return list(self.entities.values())

    def compile(self) -> Transformation:
        from aligned.compiler.feature_factory import List
        from aligned.schemas.transformation import LoadFeature

        explode_key: str | None = None
        for feature in self.entities.values():
            if isinstance(feature, List):
                explode_key = feature.name

        return LoadFeature(
            {key: value.name for key, value in self.entities.items()},
            self.feature,
            explode_key,
            dtype=self.dtype,
        )


@dataclass
class FormatString(TransformationFactory):
    format: str
    features: list[FeatureFactory]

    @property
    def using_features(self) -> list[FeatureFactory]:
        return self.features

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import FormatStringTransformation

        return FormatStringTransformation(
            self.format, [feature.name for feature in self.features]
        )


@dataclass
class ListDotProduct(TransformationFactory):
    left: FeatureFactory
    right: FeatureFactory

    @property
    def using_features(self) -> list[FeatureFactory]:
        return [self.left, self.right]

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import ListDotProduct

        return ListDotProduct(self.left.name, self.right.name)


@dataclass
class HashColumns(TransformationFactory):
    columns: list[FeatureFactory]

    @property
    def using_features(self) -> list[FeatureFactory]:
        return self.columns

    def compile(self) -> Transformation:
        from aligned.schemas.transformation import HashColumns

        return HashColumns([col.name for col in self.columns])
