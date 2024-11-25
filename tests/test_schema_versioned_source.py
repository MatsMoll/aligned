from aligned import feature_view, String, Int32, FileSource, Float
from aligned.compiler.feature_factory import EventTimestamp
from aligned.sources.random_source import RandomDataSource

source = FileSource.directory('test_data/temp').with_schema_version('version_test').parquet_at('data.parquet')


@feature_view(name='test', source=source, materialized_source=source)
class VersionOne:
    some_id = Int32().as_entity()
    feature = String()


@feature_view(name='test', source=source, materialized_source=source)
class BreakingVersion:
    some_id = Int32().as_entity()

    feature = String()
    new_feature = Int32()


@feature_view(name='test', source=source, materialized_source=source)
class AdditionalVersion:
    some_id = Int32().as_entity()

    feature = String()

    optional_value = Int32().is_optional()
    other_value = String().default_value('Hello')
    other_default = Float().default_value(0)
    contains_hello = feature.contains('Hello')


@feature_view(
    name='test',
    source=RandomDataSource(),
)
class BreakingDueToTimestamp:
    some_id = Int32().as_entity()
    feature = String()
    loaded_at = EventTimestamp()
    contains_hello = feature.contains('Hello')


def test_schema_versions() -> None:
    original = VersionOne.compile()
    compatible = AdditionalVersion.compile()
    breaking = BreakingVersion.compile()
    breaking_timestamp = BreakingDueToTimestamp.compile()

    assert original.source == compatible.source
    assert original.materialized_source == compatible.materialized_source
    assert original.materialized_source != breaking.materialized_source
    assert original.materialized_source != breaking_timestamp.materialized_source
    assert breaking.materialized_source != breaking_timestamp.materialized_source
