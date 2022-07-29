from aladdin.data_source.stream_data_source import HttpStreamSource
from aladdin.feature_store import FeatureStore
from aladdin.feature_types import (
    UUID,
    Bool,
    Contains,
    CreatedAtTimestamp,
    DateComponent,
    DifferanceBetween,
    Double,
    Entity,
    Equals,
    EventTimestamp,
    Float,
    Int32,
    Int64,
    Ratio,
    String,
    TimeDifferance,
)
from aladdin.feature_view.combined_view import CombinedFeatureView, CombinedFeatureViewMetadata
from aladdin.feature_view.feature_view import FeatureView
from aladdin.feature_view.feature_view_metadata import FeatureViewMetadata
from aladdin.local.source import FileSource
from aladdin.psql.data_source import PostgreSQLConfig
from aladdin.redis.config import RedisConfig
from aladdin.redshift.data_source import RedshiftSQLConfig
from aladdin.s3.config import AwsS3Config

try:
    import dask.dataframe as _  # noqa: F401
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)
    logger.info('Dask is not enabled, meaning .to_dask() is the same as .to_df()')

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
    'Double',
    'CreatedAtTimestamp',
    # Transformations
    'Ratio',
    'Contains',
    'Equals',
    'EventTimestamp',
    'DateComponent',
    'TimeDifferance',
    'DifferanceBetween',
]
