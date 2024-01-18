from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

import pandas as pd
import polars as pl

from aligned.compiler.vector_index_factory import VectorIndexFactory
from aligned.data_source.stream_data_source import StreamDataSource
from aligned.schemas.constraints import (
    Constraint,
    EndsWith,
    InDomain,
    LowerBoundInclusive,
    MaxLength,
    MinLength,
    Optional,
    Unique,
    Regex,
    StartsWith,
    UpperBoundInclusive,
)
from aligned.schemas.derivied_feature import DerivedFeature, AggregateOver
from aligned.schemas.event_trigger import EventTrigger as EventTriggerSchema
from aligned.schemas.feature import EventTimestamp as EventTimestampFeature
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReferance, FeatureType
from aligned.schemas.literal_value import LiteralValue
from aligned.schemas.target import ClassificationTarget as ClassificationTargetSchemas
from aligned.schemas.target import ClassTargetProbability
from aligned.schemas.target import RegressionTarget as RegressionTargetSchemas
from aligned.schemas.target import RecommendationTarget as RecommendationTargetSchemas
from aligned.schemas.transformation import EmbeddingModel, Transformation
from aligned.schemas.vector_storage import VectorStorage

if TYPE_CHECKING:
    from aligned.sources.s3 import AwsS3Config


class TransformationFactory:
    """
    A class that can compute the transformation logic.

    For most classes will there be no need for a factory,
    However for more advanced transformations will this be needed.

    e.g:

    StandaredScalerFactory(
        time_window=timedelta(days=30)
    )

    The batch data source will be provided when the compile method is run.
    leading to fetching a small sample, and compute the metrics needed in order to generate a
    """

    def compile(self) -> Transformation:
        raise NotImplementedError(type(self))

    @property
    def using_features(self) -> list[FeatureFactory]:
        raise NotImplementedError(type(self))


class AggregationTransformationFactory:
    def aggregate_over(
        self, group_by: list[FeatureReferance], time_columns: FeatureReferance | None
    ) -> AggregateOver:
        raise NotImplementedError(type(self))


T = TypeVar('T')


@dataclass
class EventTrigger:
    condition: FeatureFactory
    event: StreamDataSource


@dataclass
class TargetProbability:
    of_value: Any
    target: ClassificationLabel
    _name: str | None = None

    def __hash__(self) -> int:
        return self._name.__hash__()

    def __set_name__(self, owner, name):
        self._name = name

    def compile(self) -> ClassTargetProbability:
        return ClassTargetProbability(
            outcome=LiteralValue.from_value(self.of_value),
            feature=Feature(self._name, dtype=FeatureType.float()),
        )


class FeatureReferencable:
    def feature_referance(self) -> FeatureReferance:
        pass


def compile_hidden_features(
    feature: FeatureFactory,
    location: FeatureLocation,
    hidden_features: int,
    var_name: str,
    entities: set[Feature],
):
    aggregations = []

    features = set()
    derived_features = set()

    if feature.transformation:
        # Adding features that is not stored in the view
        # e.g:
        # class SomeView(FeatureView):
        #     ...
        #     x, y = Bool(), Bool()
        #     z = (x & y) | x
        #
        # Here will (x & y)'s result be a 'hidden' feature
        feature_deps = [(feat.depth(), feat) for feat in feature.feature_dependencies()]

        # Sorting by key in order to instanciate the "core" features first
        # And then making it possible for other features to reference them
        def sort_key(x: tuple[int, FeatureFactory]) -> int:
            return x[0]

        for depth, feature_dep in sorted(feature_deps, key=sort_key):

            if not feature_dep._location:
                feature_dep._location = location

            if feature_dep._name:
                feat_dep = feature_dep.feature()
                if feat_dep in features or feat_dep in entities:
                    continue

            if depth == 0:
                # The raw value and the transformed have the same name
                if not feature_dep._name:
                    feature_dep._name = var_name
                feat_dep = feature_dep.feature()
                features.add(feat_dep)
                continue

            if not feature_dep._name:
                feature_dep._name = str(hidden_features)
                hidden_features += 1

            if isinstance(feature_dep.transformation, AggregationTransformationFactory):
                aggregations.append(feature_dep)
            else:
                feature_graph = feature_dep.compile()  # Should decide on which payload to send
                if feature_graph in derived_features:
                    continue

                derived_features.add(feature_dep.compile())

        if not feature._name:
            feature._name = 'ephemoral'
        if isinstance(feature.transformation, AggregationTransformationFactory):
            aggregations.append(feature)
        else:
            derived_features.add(feature.compile())  # Should decide on which payload to send

    return features, derived_features


@dataclass
class RecommendationTarget(FeatureReferencable):

    feature: FeatureFactory
    rank_feature: FeatureFactory | None = field(default=None)

    _name: str | None = field(default=None)
    _location: FeatureLocation | None = field(default=None)

    def __set_name__(self, owner, name):
        self._name = name

    def feature_referance(self) -> FeatureReferance:
        if not self._name:
            raise ValueError('Missing name, can not create reference')
        if not self._location:
            raise ValueError('Missing location, can not create reference')
        return FeatureReferance(self._name, self._location, self.feature.dtype)

    def estemating_rank(self, feature: FeatureFactory) -> RecommendationTarget:
        self.rank_feature = feature
        return self

    def compile(self) -> RecommendationTargetSchemas:
        self_ref = self.feature_referance()

        return RecommendationTargetSchemas(
            self.feature.feature_referance(),
            feature=self_ref.as_feature(),
            estimating_rank=self.rank_feature.feature_referance() if self.rank_feature else None,
        )


@dataclass
class RegressionLabel(FeatureReferencable):
    feature: FeatureFactory
    event_trigger: EventTrigger | None = field(default=None)
    ground_truth_event: StreamDataSource | None = field(default=None)
    _name: str | None = field(default=None)
    _location: FeatureLocation | None = field(default=None)

    def __set_name__(self, owner, name):
        self._name = name

    def feature_referance(self) -> FeatureReferance:
        if not self._name:
            raise ValueError('Missing name, can not create reference')
        if not self._location:
            raise ValueError('Missing location, can not create reference')
        return FeatureReferance(self._name, self._location, self.feature.dtype)

    def listen_to_ground_truth_event(self, stream: StreamDataSource) -> RegressionLabel:
        return RegressionLabel(
            feature=self.feature,
            event_trigger=self.event_trigger,
            ground_truth_event=stream,
        )

    def send_ground_truth_event(self, when: Bool, sink_to: StreamDataSource) -> RegressionLabel:
        assert when.dtype == FeatureType.bool(), 'A trigger needs a boolean condition'

        return RegressionLabel(
            self.feature, EventTrigger(when, sink_to), ground_truth_event=self.ground_truth_event
        )

    def compile(self) -> RegressionTargetSchemas:
        on_ground_truth_event = self.ground_truth_event
        trigger = self.event_trigger

        if self.event_trigger:
            event = self.event_trigger
            if not event.condition._name:
                event.condition._name = '0'

            trigger = EventTriggerSchema(
                event.condition.compile(), event=event.event, payload={self.feature.feature()}
            )
            if not on_ground_truth_event:
                on_ground_truth_event = event.event

        return RegressionTargetSchemas(
            self.feature.feature_referance(),
            feature=Feature(self._name, self.feature.dtype),
            on_ground_truth_event=on_ground_truth_event,
            event_trigger=trigger,
        )


@dataclass
class ClassificationLabel(FeatureReferencable):
    feature: FeatureFactory
    event_trigger: EventTrigger | None = field(default=None)
    ground_truth_event: StreamDataSource | None = field(default=None)
    _name: str | None = field(default=None)
    _location: FeatureLocation | None = field(default=None)

    def __set_name__(self, owner, name):
        self._name = name

    def feature_referance(self) -> FeatureReferance:
        if not self._name:
            raise ValueError('Missing name, can not create reference')
        if not self._location:
            raise ValueError('Missing location, can not create reference')
        return FeatureReferance(self._name, self._location, self.feature.dtype)

    def listen_to_ground_truth_event(self, stream: StreamDataSource) -> ClassificationLabel:
        return ClassificationLabel(
            feature=self.feature,
            event_trigger=self.event_trigger,
            ground_truth_event=stream,
        )

    def send_ground_truth_event(self, when: Bool, sink_to: StreamDataSource) -> ClassificationLabel:
        assert when.dtype == FeatureType.bool(), 'A trigger needs a boolean condition'

        return ClassificationLabel(self.feature, EventTrigger(when, sink_to))

    def probability_of(self, value: Any) -> TargetProbability:
        """Define a value that will be the probability of a certain target class.

        This is mainly intended to be used for classification problems with low cardinality.

        For example, if the target is a binary classification, then the probability of the target.
        being 1 can be defined by:

        >>> target.probability_of(1)

        For cases where the target have a high cardinality can `probabilities` be used instead.

        Args:
            value (Any): The class that will be the probability of the target.

        Returns:
            TargetProbability: A feature that contains the probability of the target class.
        """

        if not isinstance(value, self.feature.dtype.python_type):
            raise ValueError(
                (
                    'Probability of target is of incorrect data type. ',
                    f'Target is {self.feature.dtype}, but value is {type(value)}.',
                )
            )

        return TargetProbability(value, self)

    def compile(self) -> ClassificationTargetSchemas:
        on_ground_truth_event = self.ground_truth_event
        trigger = self.event_trigger

        if self.event_trigger:
            event = self.event_trigger
            if not event.condition._name:
                event.condition._name = '0'

            trigger = EventTriggerSchema(
                event.condition.compile(), event=event.event, payload={self.feature.feature()}
            )
            if not on_ground_truth_event:
                on_ground_truth_event = event.event

        return ClassificationTargetSchemas(
            self.feature.feature_referance(),
            feature=Feature(self._name, self.feature.dtype),
            on_ground_truth_event=on_ground_truth_event,
            event_trigger=trigger,
        )


class FeatureFactory(FeatureReferencable):
    """
    Represents the information needed to generate a feature definition

    This may contain lazely loaded information, such as then name.
    Which will be added when the feature view is compiled, as we can get the attribute name at runtime.

    The feature can still have no name, but this means that it is an unstored feature.
    It will threfore need a transformation field.

    The feature_dependencies is the features graph for the given feature.

    aka
                            x <- standard scaler <- age: Float
    x_and_y_is_equal <-
                            y: Float
    """

    _name: str | None = None
    _location: FeatureLocation | None = None
    _description: str | None = None

    transformation: TransformationFactory | None = None
    constraints: set[Constraint] | None = None

    def __set_name__(self, owner, name):
        self._name = name

    @property
    def dtype(self) -> FeatureType:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        if not self._name:
            raise ValueError('Have not been given a name yet')
        return self._name

    @property
    def depending_on_names(self) -> list[str]:
        if not self.transformation:
            return []
        return [feat._name for feat in self.transformation.using_features if feat._name]

    def feature_referance(self) -> FeatureReferance:
        if not self._location:
            raise ValueError(
                f'_location is not set for {self.name}. '
                'Therefore, making it impossible to create a referance.'
            )
        return FeatureReferance(self.name, self._location, self.dtype)

    def feature(self) -> Feature:
        return Feature(
            name=self.name,
            dtype=self.dtype,
            description=self._description,
            tags=None,
            constraints=self.constraints,
        )

    def as_classification_label(self) -> ClassificationLabel:
        return ClassificationLabel(self)

    def as_regression_label(self) -> RegressionLabel:
        return RegressionLabel(self)

    def as_recommendation_target(self) -> RecommendationTarget:
        return RecommendationTarget(self)

    def compile(self) -> DerivedFeature:

        if not self.transformation:
            raise ValueError(f'Trying to create a derived feature with no transformation, {self.name}')

        return DerivedFeature(
            name=self.name,
            dtype=self.dtype,
            depending_on={feat.feature_referance() for feat in self.transformation.using_features},
            transformation=self.transformation.compile(),
            depth=self.depth(),
            description=self._description,
            tags=None,
            constraints=self.constraints,
        )

    def depth(self) -> int:
        if not self.transformation:
            return 0

        value = 0
        for feature in self.transformation.using_features:
            value = max(feature.depth(), value)
        return value + 1

    def description(self: T, description: str) -> T:
        self._description = description  # type: ignore [attr-defined]
        return self

    def feature_dependencies(self) -> list[FeatureFactory]:
        values = []

        if not self.transformation:
            return []

        def add_values(feature: FeatureFactory) -> None:
            values.append(feature)
            if not feature.transformation:
                return
            for sub_feature in feature.transformation.using_features:
                add_values(sub_feature)

        for sub_feature in self.transformation.using_features:
            add_values(sub_feature)

        return values

    def copy_type(self: T) -> T:
        raise NotImplementedError()

    def fill_na(self: T, value: FeatureFactory | Any) -> T:
        from aligned.compiler.transformation_factory import FillMissingFactory

        instance: FeatureFactory = self.copy_type()  # type: ignore [attr-defined]
        if not isinstance(value, FeatureFactory):
            value = LiteralValue.from_value(value)

        instance.transformation = FillMissingFactory(self, value)  # type: ignore [attr-defined]
        return instance  # type: ignore [return-value]

    def transformed_using_features_pandas(
        self: T, using_features: list[FeatureFactory], transformation: Callable[[pd.DataFrame, pd.Series]]
    ) -> T:
        from aligned.compiler.transformation_factory import PandasTransformationFactory

        dtype: FeatureFactory = self.copy_type()  # type: ignore [assignment]

        dtype.transformation = PandasTransformationFactory(dtype, transformation, using_features or [self])
        return dtype  # type: ignore [return-value]

    def transform_pandas(self, transformation: Callable[[pd.DataFrame], pd.Series], as_dtype: T) -> T:
        from aligned.compiler.transformation_factory import PandasTransformationFactory

        dtype: FeatureFactory = as_dtype  # type: ignore [assignment]

        dtype.transformation = PandasTransformationFactory(dtype, transformation, [self])
        return dtype  # type: ignore [return-value]

    def transformed_using_features_polars(
        self: T,
        using_features: list[FeatureFactory],
        transformation: Callable[[pl.LazyFrame, str], pl.LazyFrame],
    ) -> T:
        from aligned.compiler.transformation_factory import PolarsTransformationFactory

        dtype: FeatureFactory = self.copy_type()  # type: ignore [assignment]
        dtype.transformation = PolarsTransformationFactory(
            dtype,
            transformation,  # type: ignore
            using_features or [self],  # type: ignore
        )
        return dtype  # type: ignore [return-value]

    def transform_polars(
        self,
        expression: pl.Expr,
        using_features: list[FeatureFactory] | None = None,
        as_dtype: T | None = None,
    ) -> T:
        from aligned.compiler.transformation_factory import PolarsTransformationFactory

        dtype: FeatureFactory = as_dtype or self.copy_type()  # type: ignore [assignment]
        dtype.transformation = PolarsTransformationFactory(dtype, expression, using_features or [self])
        return dtype  # type: ignore [return-value]

    def polars_aggregation(self, aggregation: pl.Expr, as_type: T) -> T:
        from aligned.compiler.aggregation_factory import PolarsTransformationFactoryAggregation

        value = as_type.copy_type()  # type: ignore [assignment]
        value.transformation = PolarsTransformationFactoryAggregation(as_type, aggregation, [self])

        return value

    def polars_aggregation_using_features(
        self: T,
        using_features: list[FeatureFactory],
        aggregation: pl.Expr,
    ) -> T:
        from aligned.compiler.aggregation_factory import PolarsTransformationFactoryAggregation

        value = self.copy_type()  # type: ignore [assignment]
        value.transformation = PolarsTransformationFactoryAggregation(self, aggregation, using_features)

        return value

    def is_required(self: T) -> T:
        return self

    def is_optional(self: T) -> T:
        self._add_constraint(Optional())  # type: ignore[attr-defined]
        return self

    def _add_constraint(self, constraint: Constraint) -> None:
        # The constraint should be a lazy evaluated constraint
        # Aka, a factory, as with the features.
        # Therefore making it possible to add distribution checks
        if not self.constraints:
            self.constraints = set()
        if isinstance(constraint, Constraint):
            self.constraints.add(constraint)
        else:
            raise ValueError(f'Unable to add constraint {constraint}.')

    def is_not_null(self) -> Bool:
        from aligned.compiler.transformation_factory import NotNullFactory

        instance = Bool()
        instance.transformation = NotNullFactory(self)
        return instance


class CouldBeModelVersion:
    def as_model_version(self) -> ModelVersion:
        if isinstance(self, FeatureFactory):
            return ModelVersion(self)

        raise ValueError(f'{self} is not a feature factory, and can therefore not be a model version')


class CouldBeEntityFeature:
    def as_entity(self) -> Entity:
        if isinstance(self, FeatureFactory):
            return Entity(self)

        raise ValueError(f'{self} is not a feature factory, and can therefore not be an entity')


class EquatableFeature(FeatureFactory):

    # Comparable operators
    def __eq__(self, right: FeatureFactory | Any) -> Bool:  # type: ignore[override]
        from aligned.compiler.transformation_factory import EqualsFactory

        instance = Bool()
        instance.transformation = EqualsFactory(self, right)
        return instance

    def equals(self, right: object) -> Bool:
        return self == right

    def __ne__(self, right: FeatureFactory | Any) -> Bool:  # type: ignore[override]
        from aligned.compiler.transformation_factory import NotEqualsFactory

        instance = Bool()
        instance.transformation = NotEqualsFactory(right, self)
        return instance

    def not_equals(self, right: object) -> Bool:
        return self != right

    def is_in(self, values: list[Any]) -> Bool:
        from aligned.compiler.transformation_factory import IsInFactory

        instance = Bool()
        instance.transformation = IsInFactory(self, values)
        return instance


class ComparableFeature(EquatableFeature):
    def __lt__(self, right: object) -> Bool:
        from aligned.compiler.transformation_factory import LowerThenFactory

        instance = Bool()
        instance.transformation = LowerThenFactory(right, self)
        return instance

    def __le__(self, right: float) -> Bool:
        from aligned.compiler.transformation_factory import LowerThenOrEqualFactory

        instance = Bool()
        instance.transformation = LowerThenOrEqualFactory(right, self)
        return instance

    def __gt__(self, right: object) -> Bool:
        from aligned.compiler.transformation_factory import GreaterThenFactory

        instance = Bool()
        instance.transformation = GreaterThenFactory(self, right)
        return instance

    def __ge__(self, right: object) -> Bool:
        from aligned.compiler.transformation_factory import GreaterThenOrEqualFactory

        instance = Bool()
        instance.transformation = GreaterThenOrEqualFactory(right, self)
        return instance

    def lower_bound(self: T, value: float) -> T:
        self._add_constraint(LowerBoundInclusive(value))  # type: ignore[attr-defined]
        return self

    def upper_bound(self: T, value: float) -> T:
        self._add_constraint(UpperBoundInclusive(value))  # type: ignore[attr-defined]
        return self


class ArithmeticFeature(ComparableFeature):
    def __sub__(self, other: FeatureFactory | Any) -> Float:
        from aligned.compiler.transformation_factory import DifferanceBetweenFactory, TimeDifferanceFactory

        feature = Float()
        if self.dtype == FeatureType.datetime():
            feature.transformation = TimeDifferanceFactory(self, other)
        else:
            feature.transformation = DifferanceBetweenFactory(self, other)
        return feature

    def __radd__(self, other: FeatureFactory | Any) -> Float:
        from aligned.compiler.transformation_factory import AdditionBetweenFactory

        feature = Float()
        feature.transformation = AdditionBetweenFactory(self, other)
        return feature

    def __add__(self, other: FeatureFactory | Any) -> Float:
        from aligned.compiler.transformation_factory import AdditionBetweenFactory

        feature = Float()
        feature.transformation = AdditionBetweenFactory(self, other)
        return feature

    def __truediv__(self, other: FeatureFactory | Any) -> Float:
        from aligned.compiler.transformation_factory import RatioFactory

        feature = Float()
        if isinstance(other, FeatureFactory):
            feature.transformation = RatioFactory(self, other)
        else:
            feature.transformation = RatioFactory(self, LiteralValue.from_value(other))
        return feature

    def __floordiv__(self, other: FeatureFactory | Any) -> Float:
        from aligned.compiler.transformation_factory import RatioFactory

        feature = Float()
        if isinstance(other, FeatureFactory):
            feature.transformation = RatioFactory(self, other)
        else:
            feature.transformation = RatioFactory(self, LiteralValue.from_value(other))
        return feature

    def __abs__(self) -> Int64:
        from aligned.compiler.transformation_factory import AbsoluteFactory

        feature = Int64()
        feature.transformation = AbsoluteFactory(self)
        return feature

    def __mul__(self, other: FeatureFactory | Any) -> Float:
        from aligned.compiler.transformation_factory import MultiplyFactory

        feature = Float()
        if isinstance(other, FeatureFactory):
            feature.transformation = MultiplyFactory(self, other)
        else:
            feature.transformation = MultiplyFactory(self, LiteralValue.from_value(other))
        return feature

    def __rmul__(self, other: FeatureFactory | Any) -> Float:
        from aligned.compiler.transformation_factory import MultiplyFactory

        feature = Float()
        if isinstance(other, FeatureFactory):
            feature.transformation = MultiplyFactory(self, other)
        else:
            feature.transformation = MultiplyFactory(self, LiteralValue.from_value(other))
        return feature

    def __pow__(self, other: FeatureFactory | Any) -> Float:
        from aligned.compiler.transformation_factory import PowerFactory

        feature = Float()
        feature.transformation = PowerFactory(self, other)
        return feature

    def log1p(self) -> Float:
        from aligned.compiler.transformation_factory import LogTransformFactory

        feature = Float()
        feature.transformation = LogTransformFactory(self)
        return feature

    def clip(self: T, lower_bound: float, upper_bound: float) -> T:
        from aligned.compiler.transformation_factory import ClipFactory

        feature = self.copy_type()  # type: ignore
        feature.transformation = ClipFactory(self, lower_bound, upper_bound)  # type: ignore
        return feature


class DecimalOperations(FeatureFactory):
    def __round__(self) -> Int64:
        from aligned.compiler.transformation_factory import RoundFactory

        feature = Int64()
        feature.transformation = RoundFactory(self)
        return feature

    def round(self) -> Int64:
        return self.__round__()

    def __ceil__(self) -> Int64:
        from aligned.compiler.transformation_factory import CeilFactory

        feature = Int64()
        feature.transformation = CeilFactory(self)
        return feature

    def cail(self) -> Int64:
        return self.__ceil__()

    def __floor__(self) -> Int64:
        from aligned.compiler.transformation_factory import FloorFactory

        feature = Int64()
        feature.transformation = FloorFactory(self)
        return feature

    def floor(self) -> Int64:
        return self.__floor__()


class TruncatableFeature(FeatureFactory):
    def __trunc__(self: T) -> T:
        raise NotImplementedError()


class NumberConvertableFeature(FeatureFactory):
    def as_float(self) -> Float:
        from aligned.compiler.transformation_factory import ToNumericalFactory

        feature = Float()
        feature.transformation = ToNumericalFactory(self)
        return feature

    def __int__(self) -> Int64:
        raise NotImplementedError()

    def __float__(self) -> Float:
        raise NotImplementedError()


class InvertableFeature(FeatureFactory):
    def __invert__(self) -> Bool:
        from aligned.compiler.transformation_factory import InverseFactory

        feature = Bool()
        feature.transformation = InverseFactory(self)
        return feature


class LogicalOperatableFeature(InvertableFeature):
    def __and__(self, other: Bool) -> Bool:
        from aligned.compiler.transformation_factory import AndFactory

        feature = Bool()
        feature.transformation = AndFactory(self, other)
        return feature

    def logical_and(self, other: Bool) -> Bool:
        return self & other

    def __or__(self, other: Bool) -> Bool:
        from aligned.compiler.transformation_factory import OrFactory

        feature = Bool()
        feature.transformation = OrFactory(self, other)
        return feature

    def logical_or(self, other: Bool) -> Bool:
        return self | other


class CategoricalEncodableFeature(EquatableFeature):
    def one_hot_encode(self, labels: list[str]) -> list[Bool]:
        return [self == label for label in labels]

    def ordinal_categories(self, orders: list[str]) -> Int32:
        from aligned.compiler.transformation_factory import OrdinalFactory

        feature = Int32()
        feature.transformation = OrdinalFactory(orders, self)
        return feature

    def accepted_values(self: T, values: list[str]) -> T:
        self._add_constraint(InDomain(values))  # type: ignore[attr-defined]
        return self


class DateFeature(FeatureFactory):
    def date_component(self, component: str) -> Int32:
        from aligned.compiler.transformation_factory import DateComponentFactory

        feature = Int32()
        feature.transformation = DateComponentFactory(component, self)
        return feature

    def day(self) -> Int32:
        return self.date_component('day')

    def hour(self) -> Int32:
        return self.date_component('hour')

    def second(self) -> Int32:
        return self.date_component('second')

    def minute(self) -> Int32:
        return self.date_component('minute')

    def quarter(self) -> Int32:
        return self.date_component('quarter')

    def week(self) -> Int32:
        return self.date_component('week')

    def year(self) -> Int32:
        return self.date_component('year')

    def month(self) -> Int32:
        return self.date_component('month')

    def weekday(self) -> Int32:
        return self.date_component('dayofweek')

    def day_of_year(self) -> Int32:
        return self.date_component('ordinal_day')


class Bool(EquatableFeature, LogicalOperatableFeature):
    @property
    def dtype(self) -> FeatureType:
        return FeatureType.bool()

    def copy_type(self) -> Bool:
        return Bool()


class Float(ArithmeticFeature, DecimalOperations):
    def copy_type(self) -> Float:
        return Float()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.float()

    def aggregate(self) -> ArithmeticAggregation:
        return ArithmeticAggregation(self)


class Int8(ArithmeticFeature, CouldBeEntityFeature, CouldBeModelVersion, CategoricalEncodableFeature):
    def copy_type(self) -> Int8:
        return Int8()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.int8()

    def aggregate(self) -> ArithmeticAggregation:
        return ArithmeticAggregation(self)


class Int16(ArithmeticFeature, CouldBeEntityFeature, CouldBeModelVersion, CategoricalEncodableFeature):
    def copy_type(self) -> Int16:
        return Int16()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.int16()

    def aggregate(self) -> ArithmeticAggregation:
        return ArithmeticAggregation(self)


class Int32(ArithmeticFeature, CouldBeEntityFeature, CouldBeModelVersion, CategoricalEncodableFeature):
    def copy_type(self) -> Int32:
        return Int32()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.int32()

    def aggregate(self) -> ArithmeticAggregation:
        return ArithmeticAggregation(self)


class Int64(ArithmeticFeature, CouldBeEntityFeature, CouldBeModelVersion, CategoricalEncodableFeature):
    def copy_type(self) -> Int64:
        return Int64()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.int64()

    def aggregate(self) -> ArithmeticAggregation:
        return ArithmeticAggregation(self)


class UUID(FeatureFactory, CouldBeEntityFeature):
    def copy_type(self) -> UUID:
        return UUID()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.uuid()

    def aggregate(self) -> CategoricalAggregation:
        return CategoricalAggregation(self)


class UniqueValidateable(FeatureFactory):
    def is_unique(self: T) -> T:
        self._add_constraint(Unique())
        return self


class LengthValidatable(FeatureFactory):
    def min_length(self: T, length: int) -> T:
        self._add_constraint(MinLength(length))
        return self

    def max_length(self: T, length: int) -> T:
        self._add_constraint(MaxLength(length))
        return self


class StringValidatable(FeatureFactory):
    def validate_regex(self: T, regex: str) -> T:
        self._add_constraint(Regex(regex))
        return self

    def validate_endswith(self: T, suffix: str) -> T:
        self._add_constraint(EndsWith(suffix))
        return self

    def validate_startswith(self: T, prefix: str) -> T:
        self._add_constraint(StartsWith(prefix))
        return self


class String(
    CategoricalEncodableFeature,
    NumberConvertableFeature,
    CouldBeModelVersion,
    CouldBeEntityFeature,
    LengthValidatable,
    StringValidatable,
):
    def copy_type(self) -> String:
        return String()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.string()

    def aggregate(self) -> StringAggregation:
        return StringAggregation(self)

    def split(self, pattern: str, max_splits: int | None = None) -> String:
        raise NotImplementedError()

    def replace(self, values: dict[str, str]) -> String:
        from aligned.compiler.transformation_factory import ReplaceFactory

        feature = String()
        feature.transformation = ReplaceFactory(values, self)
        return feature

    def contains(self, value: str) -> Bool:
        from aligned.compiler.transformation_factory import ContainsFactory

        feature = Bool()
        feature.transformation = ContainsFactory(value, self)
        return feature

    def sentence_vector(self, model: EmbeddingModel) -> Embedding:
        from aligned.compiler.transformation_factory import WordVectoriserFactory

        feature = Embedding()
        feature.transformation = WordVectoriserFactory(self, model)
        feature.embedding_size = model.embedding_size
        return feature

    def embedding(self, model: EmbeddingModel) -> Embedding:
        return self.sentence_vector(model)

    def append(self, other: FeatureFactory | str) -> String:
        from aligned.compiler.transformation_factory import AppendStrings

        feature = String()
        if isinstance(other, FeatureFactory):
            feature.transformation = AppendStrings(self, other)
        else:
            feature.transformation = AppendStrings(self, LiteralValue.from_value(other))
        return feature

    def prepend(self, other: FeatureFactory | str) -> String:
        from aligned.compiler.transformation_factory import AppendStrings, PrependConstString

        feature = String()
        if isinstance(other, FeatureFactory):
            feature.transformation = AppendStrings(other, self)
        else:
            feature.transformation = PrependConstString(other, self)
        return feature

    def as_presigned_aws_url(self, credentials: AwsS3Config, max_age_seconds: int | None = None) -> ImageUrl:
        from aligned.compiler.transformation_factory import PresignedAwsUrlFactory

        feature = ImageUrl()
        feature.transformation = PresignedAwsUrlFactory(credentials, self, max_age_seconds or 30)

        return feature

    def as_image_url(self) -> ImageUrl:
        image_url = ImageUrl()
        image_url.transformation = self.transformation
        return image_url


class Json(FeatureFactory):
    def copy_type(self: Json) -> Json:
        return super().copy_type()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.json()

    def json_path_value_at(self, path: str, as_type: T) -> T:
        from aligned.compiler.transformation_factory import JsonPathFactory

        feature = as_type.copy_type()
        feature.transformation = JsonPathFactory(self, path)
        return feature

    def field(self, field: str, as_type: T) -> T:
        from aligned.compiler.transformation_factory import StructFieldFactory

        feature = as_type.copy_type()
        feature.transformation = StructFieldFactory(self, field)
        return feature


class ModelVersion(FeatureFactory):

    _dtype: FeatureFactory

    @property
    def dtype(self) -> FeatureType:
        return self._dtype.dtype

    def __init__(self, dtype: FeatureFactory):
        self._dtype = dtype

    def aggregate(self) -> CategoricalAggregation:
        return CategoricalAggregation(self)


class Entity(FeatureFactory):

    _dtype: FeatureFactory

    @property
    def dtype(self) -> FeatureType:
        return self._dtype.dtype

    def __init__(self, dtype: FeatureFactory):
        self._dtype = dtype

    def aggregate(self) -> CategoricalAggregation:
        return CategoricalAggregation(self)


class Timestamp(DateFeature, ArithmeticFeature):
    @property
    def dtype(self) -> FeatureType:
        return FeatureType.datetime()


class EventTimestamp(DateFeature, ArithmeticFeature):

    ttl: timedelta | None

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.datetime()

    def __init__(self, ttl: timedelta | None = None):
        self.ttl = ttl

    def event_timestamp(self) -> EventTimestampFeature:
        return EventTimestampFeature(
            name=self.name,
            ttl=int(self.ttl.total_seconds()) if self.ttl else None,
            description=self._description,
        )


class Embedding(FeatureFactory):

    sub_type: FeatureFactory
    embedding_size: int | None = None
    indexes: list[VectorIndexFactory] | None = None

    def copy_type(self) -> Embedding:
        return Embedding()

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.embedding()

    def indexed(
        self,
        storage: VectorStorage,
        metadata: list[FeatureFactory] | None = None,
        embedding_size: int | None = None,
    ) -> Embedding:
        if self.indexes is None:
            self.indexes = []

        if not self.embedding_size:
            assert embedding_size, 'An embedding size is needed in order to create a vector index'

        self.indexes.append(
            VectorIndexFactory(
                vector_dim=self.embedding_size or embedding_size,
                metadata=metadata or [],
                storage=storage,
            )
        )
        return self


GenericFeature = TypeVar('GenericFeature', bound=FeatureFactory)


@dataclass
class List(FeatureFactory, Generic[GenericFeature]):

    sub_type: GenericFeature

    def copy_type(self) -> List:
        return List(self.sub_type.copy_type())

    @property
    def dtype(self) -> FeatureType:
        return FeatureType.array()

    def contains(self, value: Any) -> Bool:
        from aligned.compiler.transformation_factory import ArrayContainsFactory

        feature = Bool()
        feature.transformation = ArrayContainsFactory(LiteralValue.from_value(value), self)
        return feature

    def at_index(self, index: int) -> GenericFeature:
        from aligned.compiler.transformation_factory import ArrayAtIndexFactory

        feature = self.sub_type.copy_type()
        feature.transformation = ArrayAtIndexFactory(self, index)
        return feature


class ImageUrl(StringValidatable):
    @property
    def dtype(self) -> FeatureType:
        return FeatureType.string()

    def copy_type(self) -> ImageUrl:
        return ImageUrl()

    def load_image(self) -> Image:
        from aligned.compiler.transformation_factory import LoadImageFactory

        image = Image()
        image.transformation = LoadImageFactory(self)
        return image


class Image(FeatureFactory):
    @property
    def dtype(self) -> FeatureType:
        return FeatureType.array()

    def copy_type(self) -> Image:
        return Image()

    def to_grayscale(self) -> Image:
        from aligned.compiler.transformation_factory import GrayscaleImageFactory

        image = Image()
        image.transformation = GrayscaleImageFactory(self)
        return image


@dataclass
class Coordinate:

    x: ArithmeticFeature
    y: ArithmeticFeature

    def eucledian_distance(self, to: Coordinate) -> Float:
        sub = self.x - to.x
        return (sub**2 + (self.y - to.y) ** 2) ** 0.5


@dataclass
class CustomAggregation:
    def transform_polars(self, expression: pl.Expr, using_features: list[FeatureFactory], as_dtype: T) -> T:
        from aligned.compiler.transformation_factory import PolarsTransformationFactory

        dtype: FeatureFactory = as_dtype  # type: ignore [assignment]
        dtype.transformation = PolarsTransformationFactory(dtype, expression, using_features)
        return dtype  # type: ignore [return-value]


@dataclass
class StringAggregation:

    feature: String
    time_window: timedelta | None = None
    every_window: timedelta | None = None
    offset_interval: timedelta | None = None

    def over(self, time_window: timedelta) -> StringAggregation:
        self.time_window = time_window
        return self

    def every(
        self,
        weeks: float | None = None,
        days: float | None = None,
        hours: float | None = None,
        minutes: float | None = None,
        seconds: float | None = None,
    ) -> StringAggregation:
        every_interval = timedelta(
            weeks=weeks or 0, days=days or 0, hours=hours or 0, minutes=minutes or 0, seconds=seconds or 0
        )
        return StringAggregation(self.feature, self.time_window, every_interval, self.offset_interval)

    def offset(
        self,
        weeks: float | None = None,
        days: float | None = None,
        hours: float | None = None,
        minutes: float | None = None,
        seconds: float | None = None,
    ) -> StringAggregation:
        offset_interval = timedelta(
            weeks=weeks or 0, days=days or 0, hours=hours or 0, minutes=minutes or 0, seconds=seconds or 0
        )
        return StringAggregation(self.feature, self.time_window, self.every_window, offset_interval)

    def concat(self, separator: str | None = None) -> String:
        from aligned.compiler.aggregation_factory import ConcatStringsAggrigationFactory

        feature = String()
        feature.transformation = ConcatStringsAggrigationFactory(
            self.feature,
            separator=separator,
            time_window=self.time_window,
            every_interval=self.every_window,
            offset_interval=self.offset_interval,
        )
        return feature

    def count(self) -> Int32:
        from aligned.compiler.aggregation_factory import CountAggregationFactory

        feat = Int32()
        feat.transformation = CountAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_window,
            offset_interval=self.offset_interval,
        )
        return feat


@dataclass
class CategoricalAggregation:

    feature: FeatureFactory
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    def over(
        self,
        weeks: float | None = None,
        days: float | None = None,
        hours: float | None = None,
        minutes: float | None = None,
        seconds: float | None = None,
    ) -> CategoricalAggregation:
        time_window = timedelta(
            weeks=weeks or 0, days=days or 0, hours=hours or 0, minutes=minutes or 0, seconds=seconds or 0
        )
        return CategoricalAggregation(self.feature, time_window, self.every_interval, self.offset_interval)

    def every(
        self,
        weeks: float | None = None,
        days: float | None = None,
        hours: float | None = None,
        minutes: float | None = None,
        seconds: float | None = None,
    ) -> CategoricalAggregation:
        every_interval = timedelta(
            weeks=weeks or 0, days=days or 0, hours=hours or 0, minutes=minutes or 0, seconds=seconds or 0
        )
        return CategoricalAggregation(self.feature, self.time_window, every_interval, self.offset_interval)

    def offset(
        self,
        weeks: float | None = None,
        days: float | None = None,
        hours: float | None = None,
        minutes: float | None = None,
        seconds: float | None = None,
    ) -> CategoricalAggregation:
        offset_interval = timedelta(
            weeks=weeks or 0, days=days or 0, hours=hours or 0, minutes=minutes or 0, seconds=seconds or 0
        )
        return CategoricalAggregation(self.feature, self.time_window, self.every_interval, offset_interval)

    def count(self) -> Int64:
        from aligned.compiler.aggregation_factory import CountAggregationFactory

        feat = Int64()
        feat.transformation = CountAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat


@dataclass
class ArithmeticAggregation:

    feature: ArithmeticFeature
    time_window: timedelta | None = None
    every_interval: timedelta | None = None
    offset_interval: timedelta | None = None

    def over(
        self,
        weeks: float | None = None,
        days: float | None = None,
        hours: float | None = None,
        minutes: float | None = None,
        seconds: float | None = None,
    ) -> ArithmeticAggregation:
        time_window = timedelta(
            weeks=weeks or 0, days=days or 0, hours=hours or 0, minutes=minutes or 0, seconds=seconds or 0
        )
        return ArithmeticAggregation(self.feature, time_window, self.every_interval, self.offset_interval)

    def every(
        self,
        weeks: float | None = None,
        days: float | None = None,
        hours: float | None = None,
        minutes: float | None = None,
        seconds: float | None = None,
    ) -> ArithmeticAggregation:
        every_interval = timedelta(
            weeks=weeks or 0, days=days or 0, hours=hours or 0, minutes=minutes or 0, seconds=seconds or 0
        )
        return ArithmeticAggregation(self.feature, self.time_window, every_interval, self.offset_interval)

    def offset(
        self,
        weeks: float | None = None,
        days: float | None = None,
        hours: float | None = None,
        minutes: float | None = None,
        seconds: float | None = None,
    ) -> ArithmeticAggregation:
        offset_interval = timedelta(
            weeks=weeks or 0, days=days or 0, hours=hours or 0, minutes=minutes or 0, seconds=seconds or 0
        )
        return ArithmeticAggregation(self.feature, self.time_window, self.every_interval, offset_interval)

    def sum(self) -> Float:
        from aligned.compiler.aggregation_factory import SumAggregationFactory

        feat = Float()
        feat.transformation = SumAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat

    def mean(self) -> Float:
        from aligned.compiler.aggregation_factory import MeanAggregationFactory

        feat = Float()
        feat.transformation = MeanAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat

    def min(self) -> Float:
        from aligned.compiler.aggregation_factory import MinAggregationFactory

        feat = Float()
        feat.transformation = MinAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat

    def max(self) -> Float:
        from aligned.compiler.aggregation_factory import MaxAggregationFactory

        feat = Float()
        feat.transformation = MaxAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat

    def count(self) -> Int64:
        from aligned.compiler.aggregation_factory import CountAggregationFactory

        feat = Int64()
        feat.transformation = CountAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat

    def count_distinct(self) -> Int64:
        from aligned.compiler.aggregation_factory import CountDistinctAggregationFactory

        feat = Int64()
        feat.transformation = CountDistinctAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat

    def std(self) -> Float:
        from aligned.compiler.aggregation_factory import StdAggregationFactory

        feat = Float()
        feat.transformation = StdAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat

    def variance(self) -> Float:
        from aligned.compiler.aggregation_factory import VarianceAggregationFactory

        feat = Float()
        feat.transformation = VarianceAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat

    def median(self) -> Float:
        from aligned.compiler.aggregation_factory import MedianAggregationFactory

        feat = Float()
        feat.transformation = MedianAggregationFactory(
            self.feature,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat

    def percentile(self, percentile: float) -> Float:
        from aligned.compiler.aggregation_factory import PercentileAggregationFactory

        feat = Float()
        feat.transformation = PercentileAggregationFactory(
            self.feature,
            percentile=percentile,
            time_window=self.time_window,
            every_interval=self.every_interval,
            offset_interval=self.offset_interval,
        )
        return feat
