from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
from uuid import uuid4

import polars as pl
from aligned.config_value import ConfigValue, EnvironmentValue, LiteralValue
from aligned.exposed_model.interface import (
    ExposedModel,
    Feature,
    FeatureReference,
    RetrievalJob,
    VersionedModel,
)
from aligned.feature_store import ModelFeatureStore
from aligned.schemas.codable import Codable
from aligned.schemas.constraints import Constraint, InDomain, Optional
from aligned.schemas.feature import StaticFeatureTags, FeatureType
from aligned.schemas.model import Model

if TYPE_CHECKING:
    from openai import AsyncClient

logger = logging.getLogger(__name__)


def write_batch_request(texts: list[str], path: Path, model: str, url: str) -> None:
    """
    Creates a .jsonl file for batch processing, with each line being a request to the embeddings API.
    """
    with path.open("w") as f:
        for i, text in enumerate(texts):
            request = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": url,
                "body": {"model": model, "input": text},
            }
            f.write(json.dumps(request) + "\n")


async def chunk_batch_embedding_request(
    texts: list[str], model: str, client: AsyncClient
) -> pl.DataFrame:
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


async def make_batch_embedding_request(
    texts: list[str], model: str, client: AsyncClient
) -> pl.DataFrame:
    id_path = str(uuid4())
    batch_file = Path(id_path)
    output_file = Path(id_path + "-output.jsonl")

    write_batch_request(texts, batch_file, model, "/v1/embeddings")
    request_file = await client.files.create(file=batch_file, purpose="batch")
    response = await client.batches.create(
        input_file_id=request_file.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={"description": "Embedding batch job"},
    )
    status_response = await client.batches.retrieve(response.id)

    last_process = None
    expected_duration_left = 60

    while status_response.status not in ["completed", "failed"]:
        await asyncio.sleep(expected_duration_left * 0.8)  # Poll every minute
        status_response = await client.batches.retrieve(response.id)
        logger.info(f"Status of batch request {status_response.status}")

        processed_records = 0
        leftover_records = 0

        if status_response.request_counts:
            processed_records = status_response.request_counts.completed
            leftover_records = status_response.request_counts.total - processed_records

        if status_response.in_progress_at:
            last_process = datetime.fromtimestamp(
                status_response.in_progress_at, tz=timezone.utc
            )
            now = datetime.now(tz=timezone.utc)

            items_per_process = (now - last_process).total_seconds() / max(
                processed_records, 1
            )
            expected_duration_left = max(items_per_process * leftover_records, 60)

    batch_info = await client.batches.retrieve(response.id)
    output_file_id = batch_info.output_file_id

    if not output_file_id:
        raise ValueError(f"No output file for request: {response.id}")

    output_content = await client.files.content(output_file_id)
    output_file.write_text(output_content.text)
    embeddings = pl.read_ndjson(output_file.as_posix())
    expanded_emb = (
        embeddings.unnest("response")
        .unnest("body")
        .explode("data")
        .select(["custom_id", "data"])
        .unnest("data")
        .select(["custom_id", "embedding"])
        .with_columns(pl.col("custom_id").str.split("-").list.get(1).alias("index"))
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
    text_length = len(texts)

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
        logger.info(f"Embedded {last_chunk_index + 1} texts out of {text_length}.")

    return embeddings


@dataclass
class OpenAiConfig(Codable):
    api_key: ConfigValue = field(
        default_factory=lambda: EnvironmentValue(env="OPENAI_API_KEY")
    )
    base_url: ConfigValue = field(
        default_factory=lambda: LiteralValue("https://api.openai.com/v1/")
    )

    def client(self) -> AsyncClient:
        from openai import AsyncClient

        return AsyncClient(api_key=self.api_key.read(), base_url=self.base_url.read())


@dataclass
class OpenAiEmbeddingPredictor(ExposedModel, VersionedModel):
    model: str
    config: OpenAiConfig

    batch_on_n_chunks: int | None = field(default=100)
    feature_refs: list[FeatureReference] = field(default=None)  # type: ignore
    output_name: str = field(default="")
    prompt_template: str = field(default="")

    model_type: str = field(default="openai_emb")

    @property
    def base_url(self) -> str:
        return self.config.base_url.read()

    @property
    def exposed_at_url(self) -> str | None:
        return self.base_url

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
        return store.store.requests_for_features(
            self.feature_refs
        ).request_result.entities

    def with_contract(self, model: Model) -> ExposedModel:
        embeddings = model.predictions_view.embeddings()
        assert (
            len(embeddings) == 1
        ), f"Need at least one embedding. Got {len(embeddings)}"

        if self.output_name == "":
            self.output_name = embeddings[0].name

        if not self.feature_refs:
            self.feature_refs = list(model.feature_references())

        if self.prompt_template == "":
            for feat in self.feature_refs:
                self.prompt_template += f"{feat.name}: {{{feat.name}}}\n\n"

        return self

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        client = self.config.client()

        expected_cols = {feat.name for feat in self.feature_refs}
        missing_cols = expected_cols - set(values.loaded_columns)

        if missing_cols:
            logging.info(f"Missing cols: {missing_cols}")
            df = await store.store.features_for(
                values, features=self.feature_refs
            ).to_polars()
        else:
            df = await values.to_polars()

        if len(expected_cols) == 1:
            texts = df[self.feature_refs[0].name].to_list()
        else:
            texts: list[str] = []
            for row in df.to_dicts():
                texts.append(self.prompt_template.format(**row))

        realtime_emb = await embed_texts(
            texts,
            model=self.model,
            skip_if_n_chunks=self.batch_on_n_chunks,
            client=client,
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

        return df.hstack([batch_result["embedding"].alias(self.output_name)])


@dataclass
class OpenAiPromptModel(ExposedModel, VersionedModel):
    model: str
    config: OpenAiConfig

    feature_refs: list[FeatureReference] = field(default=None)  # type: ignore
    output_name: str = field(default="")
    prompt_template: str = field(default="")
    model_type: str = field(default="openai_completion")

    @property
    def base_url(self) -> str:
        return self.config.base_url.read()

    @property
    def exposed_at_url(self) -> str | None:
        return self.base_url

    def prompt_template_hash(self) -> str:
        from hashlib import sha256

        return sha256(self.prompt_template.encode(), usedforsecurity=False).hexdigest()

    @property
    def as_markdown(self) -> str:
        return f"""Sending a `completion` request to OpenAI's API.

This will use the model: `{self.model}` to generate the a completion.

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
        return store.store.requests_for_features(
            self.feature_refs
        ).request_result.entities

    def with_contract(self, model: Model) -> ExposedModel:
        if self.output_name == "":
            completion = model.predictions_view.prompt_completion_feature
            assert completion
            self.output_name = completion.name

        if not self.feature_refs:
            self.feature_refs = list(model.feature_references())

        if self.prompt_template == "":
            for feat in self.feature_refs:
                self.prompt_template += f"{feat.name}: {{{feat.name}}}\n\n"

        return self

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        client = self.config.client()

        expected_cols = {feat.name for feat in self.feature_refs}
        missing_cols = expected_cols - set(values.loaded_columns)

        if missing_cols:
            logging.info(f"Missing cols: {missing_cols}")
            df = await store.store.features_for(
                values, features=self.feature_refs
            ).to_polars()
        else:
            df = await values.to_polars()

        if len(expected_cols) == 1:
            texts = df[self.feature_refs[0].name].to_list()
        else:
            texts: list[str] = []
            for row in df.to_dicts():
                texts.append(self.prompt_template.format(**row))

        text_length = len(texts)
        responses: list[str | None] = []
        for index, prompt in enumerate(texts):
            res = await client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            responses.append(res.choices[0].message.content)
            logger.info(f"Prompted {index + 1} out of {text_length} prompts.")

        return df.hstack(
            [pl.Series(name=self.output_name, values=responses, dtype=pl.String())]
        )


@dataclass
class OpenAiExtractModel(ExposedModel, VersionedModel):
    model: str
    extract_task_description: str
    config: OpenAiConfig

    feature_refs: list[FeatureReference] = field(default=None)  # type: ignore
    output_name: str = field(default="")
    extract_features: list[Feature] = field(default_factory=list)

    model_type: str = field(default="openai_extract")

    @staticmethod
    def default_extract_message() -> str:
        return "Extract as many features as possible based on the expected output schema. Do not explain why or how you found the value, only return the value."

    def input_template(self, refs: list[Feature]) -> str:
        return "\n".join([f"<{ref.name}>{{{ref.name}}}</{ref.name}>" for ref in refs])

    def output_schema_description(self) -> str:
        json_format = "{\n"

        for feat in self.extract_features:
            feature_description = ""
            if feat.description:
                feature_description += f"// {feat.description}\n"
            if feat.constraints:
                feature_description += "// We assume that the feature will have the following constraints\n"
                for constraint in feat.constraints:
                    feature_description += f"// {constraint.description()}\n"
            feature_description += f'"{feat.name}": "{feat.dtype.name}",\n'
            json_format += feature_description

        return json_format + "}"

    @property
    def base_url(self) -> str:
        return self.config.base_url.read()

    @property
    def exposed_at_url(self) -> str | None:
        return self.base_url

    def prompt_template_hash(self) -> str:
        from hashlib import sha256

        return sha256(
            self.extract_task_description.encode(), usedforsecurity=False
        ).hexdigest()

    @property
    def as_markdown(self) -> str:
        return f"""Extracts the following features:
{[feat.name for feat in self.extract_features]}

Based on the feature(s): {[feat.name for feat in self.feature_refs]}

By sending a request to a the model '{self.model}' using the config {self.config}."""

    async def model_version(self) -> str:
        if len(self.feature_refs) == 1:
            return f"{self.model}-{self.feature_refs[0].identifier}"
        else:
            return f"{self.model}-{self.prompt_template_hash()}"

    async def needed_features(self, store: ModelFeatureStore) -> list[FeatureReference]:
        return self.feature_refs

    async def needed_entities(self, store: ModelFeatureStore) -> set[Feature]:
        return store.store.requests_for_features(
            self.feature_refs
        ).request_result.entities

    def with_contract(self, model: Model) -> ExposedModel:
        if self.output_name == "":
            completion = model.predictions_view.prompt_completion_feature
            self.output_name = completion.name if completion else "prompt_output"

        if not self.feature_refs:
            self.feature_refs = list(model.feature_references())

        if (
            self.extract_task_description
            == OpenAiExtractModel.default_extract_message()
        ):
            self.extract_task_description += f"You are extracting a '{model.name}'\n"

        if not self.extract_features:
            ignore_tags = [
                StaticFeatureTags.is_prompt_completion,
                StaticFeatureTags.is_annotated_by,
                StaticFeatureTags.is_model_version,
                StaticFeatureTags.is_shadow_model,
                StaticFeatureTags.is_entity,
                StaticFeatureTags.is_freshness,
            ]
            for feature in model.predictions_view.features:
                should_skip = False
                if feature.tags:
                    for ignore_tag in ignore_tags:
                        if ignore_tag in feature.tags:
                            should_skip = True
                            break

                if not should_skip:
                    self.extract_features.append(feature)

        return self

    async def run_polars(
        self, values: RetrievalJob, store: ModelFeatureStore
    ) -> pl.DataFrame:
        client = self.config.client()

        expected_cols = {feat.name for feat in self.feature_refs}
        missing_cols = expected_cols - set(values.loaded_columns)

        pred_view = store.model.predictions_view

        pred_at = pred_view.freshness_feature
        model_version_feat = pred_view.model_version_column

        job = store.store.features_for(values, features=self.feature_refs)

        if missing_cols:
            logger.info(f"Missing cols: {missing_cols}")
            df = await job.to_polars()
        else:
            df = await values.to_polars()

        if model_version_feat:
            df = df.with_columns(
                pl.lit(await self.model_version()).alias(model_version_feat.name)
            )

        messages: list[dict[str, Any]] = []

        texts: list[str] = []
        output_format = self.output_schema_description()

        image_features: list[Feature] = []
        input_features: list[Feature] = []

        for feature in job.request_result.features:
            if feature.name not in expected_cols:
                continue

            if feature.tags and StaticFeatureTags.is_image in feature.tags:
                image_features.append(feature)
            else:
                input_features.append(feature)

        input_template = self.input_template(input_features)

        def image_format_from(image: bytes) -> str:
            img_types = {
                b"\xff\xd8\xff\xdb": "jpeg",
                b"\xff\xd8\xff\xe0": "jpeg",
                b"\xff\xd8\xff\xee": "jpeg",
                b"\xff\xd8\xff\xe1": "jpeg",
                b"\x47\x49\x46\x38\x37\x61": "gif",
                b"\x47\x49\x46\x38\x39\x61": "gif",
                b"\x42\x4d": "bmp",
                b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            }

            for prefix, image_format in img_types.items():
                if image.startswith(prefix):
                    return image_format

            return "jpeg"

        for row in df.to_dicts():
            message = f"{self.extract_task_description}\n{input_template.format(**row)}\nOutput format should be in JSON with the following schema and description\n{output_format}. If you are unsure return 'null' or exclude the field."

            message = f"{self.extract_task_description}\n{input_template.format(**row)}.\nIf you are unsure about a feature, then do not return anything for that feature."

            if image_features:
                import base64

                message_content: list[dict[str, Any]] = [
                    {"type": "text", "text": message}
                ]

                for feat in image_features:
                    if feat.dtype == FeatureType.string():
                        message_content.append(
                            {"type": "image_url", "image_url": {"url": row[feat.name]}}
                        )
                    else:
                        image_bytes = row[feat.name]
                        im_format = image_format_from(image_bytes)
                        base_encoded = base64.b64encode(image_bytes)
                        message_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{im_format};base64,{base_encoded.decode()}",
                                },
                            }
                        )

                messages.append({"role": "user", "content": message_content})
            else:
                messages.append({"role": "user", "content": message})

        def schema_for_dtype(
            dtype: FeatureType,
            description: str | None = None,
            constraints: Iterable[Constraint] | None = None,
        ) -> dict:
            if dtype.name == FeatureType.boolean().name:
                ret = {"type": "boolean"}
                if description:
                    ret["description"] = description
                return ret

            if dtype.is_numeric:
                ret = {"type": "number"}
                if description:
                    ret["description"] = description
                return ret

            if dtype.is_array:
                sub_type = dtype.array_subtype() or FeatureType.string()

                ret = {
                    "type": "array",
                    "items": schema_for_dtype(sub_type),
                }

                if description:
                    ret["description"] = description

                return ret

            if dtype.is_struct:
                fields = dtype.struct_fields()

                return {
                    "type": "object",
                    "parameters": {
                        name: schema_for_dtype(dtype) for name, dtype in fields.items()
                    },
                }

            ret: dict[str, Any] = {"type": dtype.name}
            if description:
                ret["description"] = description

            if constraints:
                for constraint in constraints:
                    if isinstance(constraint, InDomain):
                        ret["enum"] = constraint.values
            return ret

        text_length = len(texts)
        responses: list[dict[str, Any] | None] = []

        for index, message in enumerate(messages):
            try:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "description": "The response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                feat.name: schema_for_dtype(
                                    feat.dtype, feat.description, feat.constraints
                                )
                                for feat in self.extract_features
                            },
                            "required": [
                                feat.name
                                for feat in self.extract_features
                                if feat.constraints is None
                                or Optional() not in feat.constraints
                            ],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
                logger.info(response_format)

                res = await client.chat.completions.create(
                    model=self.model,
                    messages=[message],  # type: ignore
                    response_format=response_format,  # type: ignore
                )
                content = res.choices[0].message.content
                if content is None:
                    continue

                decoded = json.loads(content)
                decoded[self.output_name] = content
                if pred_at:
                    decoded[pred_at.name] = datetime.now(timezone.utc)

                responses.append(decoded)

                logger.info(f"Prompted {index + 1} out of {text_length} prompts.")
            except Exception as e:
                logger.exception(e)
                responses.append({"error": str(e)})

        return df.hstack(pl.DataFrame(responses))
