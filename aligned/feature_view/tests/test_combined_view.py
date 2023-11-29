import pytest

from aligned import FeatureStore, feature_view, Int32, FileSource


@pytest.mark.asyncio
async def test_combined_view(combined_feature_store: FeatureStore) -> None:

    entities = {'passenger_id': [1, 2, 3, 4, None], 'scan_id': [842302, 84300903, 843786, None, 842301]}
    result_job = combined_feature_store.features_for(
        entities,
        features=[
            'combined:some_feature',
            'combined:other_feature',
        ],
    )
    result = await result_job.log_each_job().to_pandas()

    assert 'some_feature' in result.columns
    assert 'other_feature' in result.columns

    assert result.shape == (len(entities['passenger_id']), 4)
    assert result.isna().sum().sum() == 4 + 2


@pytest.mark.asyncio
async def test_combined_view_get_all_features(combined_feature_store: FeatureStore) -> None:

    entities = {'passenger_id': [1, 2, 3, 4, None], 'scan_id': [842302, 84300903, 843786, None, 842301]}
    result = await combined_feature_store.features_for(entities, features=['combined:*']).to_pandas()

    assert 'some_feature' in result.columns
    assert 'other_feature' in result.columns

    assert result.shape == (len(entities['passenger_id']), 4)
    assert result.isna().sum().sum() == 4 + 2


@pytest.mark.asyncio
async def test_new_combined_solution() -> None:
    import pandas as pd

    expected_df = pd.DataFrame({'other_id': [6, 5], 'new_feature': [600, 400], 'some_id': [1, 2]})

    @feature_view(name='test', source=FileSource.csv_at('test_data/test.csv'))
    class Test:
        some_id = Int32().as_entity()

        feature = Int32()

        derived_feature = feature * 10

    @feature_view(name='other', source=FileSource.csv_at('test_data/other.csv'))
    class Other:

        other_id = Int32().as_entity()
        some_id = Int32()

        other_feature = Int32()

        test_feature = other_feature * 10

    test = Test()
    other = Other()

    @feature_view(name='combined', source=Test.join(other, on=test.some_id))  # type: ignore
    class Combined:
        some_id = Int32().as_entity()

        new_feature = test.derived_feature * other.test_feature

    result = await Combined.query().all().to_pandas()  # type: ignore
    result['new_feature'] = result['new_feature'].astype('int64')
    assert result[expected_df.columns].equals(expected_df)


@pytest.mark.asyncio
async def test_view_reference() -> None:
    import pandas as pd

    expected_df = pd.DataFrame({'new_feature': [100, 100, 100], 'some_id': [1, 2, 3]})

    @feature_view(name='test', source=FileSource.csv_at('test_data/test.csv'))
    class Test:
        some_id = Int32().as_entity()

        feature = Int32()

        derived_feature = feature * 10

    test = Test()

    @feature_view(name='test_ref', source=Test)  # type: ignore
    class TestRef:
        some_id = Int32().as_entity()

        new_feature = test.derived_feature * 5

    result = await TestRef.query().all().to_pandas()  # type: ignore
    result['new_feature'] = result['new_feature'].astype('int64')
    assert result[expected_df.columns].equals(expected_df)
