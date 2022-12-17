"""
.. include:: ../README.md
"""
from aligned.compiler.feature_factory import (
    UUID,
    Bool,
    Entity,
    EventTimestamp,
    Float,
    Int32,
    Int64,
    String,
    Timestamp,
)
from aligned.data_source.stream_data_source import HttpStreamSource
from aligned.feature_store import FeatureStore
from aligned.feature_view import (
    CombinedFeatureView,
    CombinedFeatureViewMetadata,
    FeatureView,
    FeatureViewMetadata,
)
from aligned.local.source import FileSource
from aligned.psql.data_source import PostgreSQLConfig
from aligned.redis.config import RedisConfig
from aligned.redshift.data_source import RedshiftSQLConfig
from aligned.s3.config import AwsS3Config

__all__ = [
    'FeatureStore',
    'FeatureView',
    'FeatureViewMetadata',
    'CombinedFeatureView',
    'CombinedFeatureViewMetadata',
    # Batch Data sources
    'PostgreSQLConfig',
    'FileSource',
    'AwsS3Config',
    'RedshiftSQLConfig',
    # Stream Data Source
    'HttpStreamSource',
    # Online Source
    'RedisConfig',
    # Types
    'Entity',
    'String',
    'Bool',
    'Entity',
    'UUID',
    'Int32',
    'Int64',
    'Float',
    'EventTimestamp',
    'Timestamp',
]
