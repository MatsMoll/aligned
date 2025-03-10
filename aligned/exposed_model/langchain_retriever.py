import polars as pl

from aligned.exposed_model.interface import PromptModel
from aligned.feature_store import ContractStore
from aligned.lazy_imports import langchain_core


class AlignedRetriever(langchain_core.retriveres.BaseRetriever):
    store: ContractStore
    index_name: str
    number_of_docs: int

    def __str__(self) -> str:
        return (
            f"Aligned Retriver - Loading {self.number_of_docs} from '{self.index_name}'"
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "langchain_core.callbacks.manager.CallbackManagerForRetrieverRun",
    ) -> list["langchain_core.documents.base.Document"]:
        raise NotImplementedError()

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: "langchain_core.callbacks.manager.AsyncCallbackManagerForRetrieverRun",
    ) -> list["langchain_core.documents.base.Document"]:
        store = self.store
        index = store.vector_index(self.index_name)
        embed_model = store.model(index.model.name)

        assert embed_model.has_exposed_model(), f"The model {index.model.name} do not have an exposed model. Which means we can not use it."

        exposed_model = embed_model.model.exposed_model

        if (
            isinstance(exposed_model, PromptModel)
            and exposed_model.precomputed_prompt_key
        ):
            input_name = exposed_model.precomputed_prompt_key
        else:
            inputs = list(embed_model.model.feature_references())
            assert len(inputs) == 1, (
                f"Model have more than one inputs: {len(inputs)}. "
                f"Unclear what to name the query: '{query}'. "
                "This can be fixed by making sure the underlying model is a "
                "`PromptModel` with a `precomputed_prompt_key`."
            )
            input_name = inputs[0].name

        embedding = (
            await store.model(embed_model.model.name)
            .predict_over({input_name: [query]})
            .to_polars()
        )

        embedding_output = [
            feature.name
            for feature in embed_model.prediction_request().all_returned_features
            if not feature.dtype.is_embedding or feature.dtype.is_array
        ]

        documents = await index.nearest_n_to(
            entities=embedding.select(pl.exclude(input_name)),
            number_of_records=self.number_of_docs,
        ).to_polars()

        documents = documents.with_columns(
            page_content=pl.concat_str(
                [pl.col(col).cast(pl.String) for col in embedding_output],
                separator="\n\n",
            )
        )

        return [
            langchain_core.documents.base.Document(**doc)
            for doc in documents.to_dicts()
        ]
