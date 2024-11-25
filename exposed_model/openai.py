from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import polars as pl
from aligned.exposed_model.interface import (
    ExposedModel,
    Feature,
    FeatureReference,
    RetrivalJob,
    VersionedModel,
)
from aligned.feature_store import ModelFeatureStore
from aligned.schemas.model import Model

if TYPE_CHECKING:
    from openai import AsyncClient

logger = logging.getLogger(__name__)


def write_batch_request(texts: list[str], path: Path, model: str, url: str) -> None:
    """
    Creates a .jsonl file for batch processing, with each line being a request to the embeddings API.
    """
    with path.open('w') as f:
        for i, text in enumerate(texts):
            request = {
                'custom_id': f"request-{i+1}",
                'method': 'POST',
                'url': url,
                'body': {'model': model, 'input': text},
            }
            f.write(json.dumps(request) + '\n')


async def chunk_batch_embedding_request(texts: list[str], model: str, client: AsyncClient) -> pl.DataFrame:

    max_batch = 50_000
    number_of_batches = ceil(len(texts) / max_batch)

    batch_result: pl.DataFrame | None = None

    for i in range(number_of_batches):
        start = i * max_batch
        end_batch = min((i + 1) * max_batch, len(texts))

        if start == end_batch:
            batch_prompts = [texts[start]]
        else:
            batch_prompts = texts[start:end_batch]

        result = await make_batch_embedding_request(batch_prompts, model, client)

        if batch_result is None:
            batch_result = result
        else:
            batch_result = batch_result.hstack(result)

    assert batch_result is not None
    return batch_result


async def make_batch_embedding_request(texts: list[str], model: str, client: AsyncClient) -> pl.DataFrame:

    id_path = str(uuid4())
    batch_file = Path(id_path)
    output_file = Path(id_path + '-output.jsonl')

    write_batch_request(texts, batch_file, model, '/v1/embeddings')
    request_file = await client.files.create(file=batch_file, purpose='batch')
    response = await client.batches.create(
        input_file_id=request_file.id,
        endpoint='/v1/embeddings',
        completion_window='24h',
        metadata={'description': 'Embedding batch job'},
    )
    status_response = await client.batches.retrieve(response.id)

    last_process = None
    expected_duration_left = 60

    while status_response.status not in ['completed', 'failed']:
        await asyncio.sleep(expected_duration_left * 0.8)  # Poll every minute
        status_response = await client.batches.retrieve(response.id)
        logger.info(f"Status of batch request {status_response.status}")

        processed_records = 0
        leftover_records = 0

        if status_response.request_counts:
            processed_records = status_response.request_counts.completed
            leftover_records = status_response.request_counts.total - processed_records

        if status_response.in_progress_at:
            last_process = datetime.fromtimestamp(status_response.in_progress_at, tz=timezone.utc)
            now = datetime.now(tz=timezone.utc)

            items_per_process = (now - last_process).total_seconds() / max(processed_records, 1)
            expected_duration_left = max(items_per_process * leftover_records, 60)

    batch_info = await client.batches.retrieve(response.id)
    output_file_id = batch_info.output_file_id

    if not output_file_id:
        raise ValueError(f"No output file for request: {response.id}")

    output_content = await client.files.retrieve_content(output_file_id)
    output_file.write_text(output_content)
    embeddings = pl.read_ndjson(output_file.as_posix())
    expanded_emb = (
        embeddings.unnest('response')
        .unnest('body')
        .explode('data')
        .select(['custom_id', 'data'])
        .unnest('data')
        .select(['custom_id', 'embedding'])
        .with_columns(pl.col('custom_id').str.split('-').list.get(1).alias('index'))
    )
    return expanded_emb


async def embed_texts(
    texts: list[str], model: str, skip_if_n_chunks: int | None, client: AsyncClient
) -> list[list[float]] | str:
    import tiktoken

    max_token_size = 8192
    number_of_texts = len(texts)

    chunks: list[int] = []
    chunk_size = 0
    encoder = tiktoken.encoding_for_model(model)

    for index, text in enumerate(texts):
        token_size = len(encoder.encode(text))

        if chunk_size + token_size > max_token_size:
            chunks.append(index)
            chunk_size = 0

            if skip_if_n_chunks and len(chunks) + 1 >= skip_if_n_chunks:
                return f"At text nr: {index} did it go above {skip_if_n_chunks} with {len(chunks)}"

        chunk_size += token_size

    if not chunks or number_of_texts - 1 > chunks[-1]:
        chunks.append(number_of_texts - 1)

    embeddings: list[list[float]] = []

    last_chunk_index = 0

    for chunk_index in chunks:
        if last_chunk_index == 0 and chunk_index >= number_of_texts - 1:
            chunk_texts = texts
        elif last_chunk_index == 0:
            chunk_texts = texts[:chunk_index]
        elif chunk_index >= number_of_texts - 1:
            chunk_texts = texts[last_chunk_index:]
        else:
            chunk_texts = texts[last_chunk_index:chunk_index]

        res = await client.embeddings.create(input=chunk_texts, model=model)
        embeddings.extend([emb.embedding for emb in res.data])
        last_chunk_index = chunk_index

    return embeddings


@dataclass
class OpenAiEmbeddingPredictor(ExposedModel, VersionedModel):

    model: str

    batch_on_n_chunks: int | None = 100
    feature_refs: list[FeatureReference] = None  # type: ignore
    output_name: str = ''
    prompt_template: str = ''
    model_type = 'openai_emb'

    @property
    def exposed_at_url(self) -> str | None:
        return 'https://api.openai.com/'

    def prompt_template_hash(self) -> str:
        from hashlib import sha256

        return sha256(self.prompt_template.encode(), usedforsecurity=False).hexdigest()

    @property
    def as_markdown(self) -> str:
        return f"""Sending a `embedding` request to OpenAI's API.

This will use the model: `{self.model}` to generate the embeddings.
Will switch to the batch API if more then {self.batch_on_n_chunks} chunks are needed to fulfill the request.

And use the prompt template:
```
{self.prompt_template}
```"""

    async def model_version(self) -> str:
        if len(self.feature_refs) == 1:
            return self.model
        else:
            return f"{self.model}-{self.prompt_template_hash()}"

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return self.feature_refs

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.store.requests_for_features(self.feature_refs).request_result.entities

    def with_contract(self, model: Model) -> ExposedModel:
        embeddings = model.predictions_view.embeddings()
        assert len(embeddings) == 1, f"Need at least one embedding. Got {len(embeddings)}"

        if self.output_name == '':
            self.output_name = embeddings[0].name

        if not self.feature_refs:
            self.feature_refs = list(model.feature_references())

        if self.prompt_template == '':
            for feat in self.feature_refs:
                self.prompt_template += f"{feat.name}: {{{feat.name}}}\n\n"

        return self

    async def run_polars(self, values: RetrivalJob, store: ModelFeatureStore) -> pl.DataFrame:
        from openai import AsyncClient

        client = AsyncClient()

        expected_cols = {feat.name for feat in self.feature_refs}
        missing_cols = expected_cols - set(values.loaded_columns)

        if missing_cols:
            logging.info(f"Missing cols: {missing_cols}")
            df = await store.store.features_for(values, features=self.feature_refs).to_polars()
        else:
            df = await values.to_polars()

        if len(expected_cols) == 1:
            texts = df[self.feature_refs[0].name].to_list()
        else:
            texts: list[str] = []
            for row in df.to_dicts():
                texts.append(self.prompt_template.format(**row))

        realtime_emb = await embed_texts(
            texts, model=self.model, skip_if_n_chunks=self.batch_on_n_chunks, client=client
        )

        if isinstance(realtime_emb, list):
            return df.hstack(
                [
                    pl.Series(
                        name=self.output_name,
                        values=realtime_emb,
                        dtype=pl.List(pl.Float32),
                    )
                ]
            )

        batch_result = await chunk_batch_embedding_request(texts, self.model, client)

        return df.hstack([batch_result['embedding'].alias(self.output_name)])
