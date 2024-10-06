import polars as pl
from aligned import model_contract, feature_view, String, Embedding, ExposedModel
from aligned.feature_store import ContractStore, ModelFeatureStore
from aligned.sources.lancedb import LanceDBConfig
from aligned.sources.random_source import RandomDataSource

import pytest


@pytest.mark.asyncio
async def test_lancedb() -> None:
    table = 'my_embedding'
    config = LanceDBConfig('test_data/temp/lancedb')

    @feature_view(source=RandomDataSource())
    class TestFeatureView:
        id = String().as_entity()
        text = String()

    async def predictor(texts: pl.DataFrame, store: ModelFeatureStore) -> pl.DataFrame:

        embeddings = []
        for text in texts['text'].to_list():
            embeddings.append([1.0 * len(text), 2.0 * len(text)])

        return texts.hstack([pl.Series(name='embedding', values=embeddings)])

    @model_contract(
        name='my_embedding',
        input_features=[TestFeatureView().text],
        output_source=config.table(table),
        exposed_model=ExposedModel.polars_predictor(predictor),
    )
    class MyEmbedding:
        id = String().as_entity()
        embedding = Embedding(embedding_size=2)

    schema = MyEmbedding.compile().predictions_view.request('').pyarrow_schema()
    conn = await config.connect()
    await conn.create_table(table, schema=schema, mode='overwrite')

    store = ContractStore.empty()
    store.add_view(TestFeatureView)
    store.add_model(MyEmbedding)

    await store.model('my_embedding').predict_over(
        {'text': ['a', 'bc'], 'id': [1, 2]}
    ).insert_into_output_source()

    df = (
        await store.vector_index('my_embedding')
        .nearest_n_to(
            entities={
                'text': ['a', 'abcd'],
            },
            number_of_records=2,
        )
        .to_polars()
    )

    assert df.height == 4
    assert 'text' in df.columns

    df = (
        await store.vector_index('my_embedding')
        .nearest_n_to(
            entities={
                'embedding': [[1.0, 2.0], [2.0, 4.0]],
            },
            number_of_records=2,
        )
        .to_polars()
    )

    assert df.height == 4
    assert 'text' not in df.columns
