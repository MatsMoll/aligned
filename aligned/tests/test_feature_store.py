from aligned import ContractStore
from aligned.sources.local import ParquetFileSource
from aligned.schemas.feature import FeatureLocation


def test_update_source(titanic_feature_store: ContractStore) -> None:

    sources = []

    def update_parquet_source(source: ParquetFileSource, loc: FeatureLocation) -> None:
        source.path = 'updated.parquet'
        sources.append(loc)

    titanic_feature_store.sources_of_type(ParquetFileSource, update_parquet_source)

    assert sources
    assert sources == [FeatureLocation.feature_view('titanic_parquet')]
