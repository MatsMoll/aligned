import logging
from collections.abc import Iterable
from time import monotonic
from typing import Any, overload

from aligned import ContractStore, FeatureLocation
from aligned.schemas.constraints import InDomain
from aligned.schemas.feature import Constraint, FeatureType

from aligned.lazy_imports import fastapi

logger = logging.getLogger(__name__)


def schema_for_dtype(
    dtype: FeatureType,
    description: str | None = None,
    constraints: Iterable[Constraint] | None = None,
) -> dict:
    if dtype.is_numeric:
        return {"type": "number", "description": description}

    if dtype.is_array:
        sub_type = dtype.array_subtype() or FeatureType.string()
        return {
            "type": "array",
            "description": description,
            "items": schema_for_dtype(sub_type),
        }

    if dtype.is_struct:
        fields = dtype.struct_fields()

        return {
            "type": "object",
            "parameters": {
                name: schema_for_dtype(dtype) for name, dtype in fields.items()
            },
        }

    ret = {"type": dtype.name, "description": description}
    if constraints:
        for constraint in constraints:
            if isinstance(constraint, InDomain):
                ret["enum"] = constraint.values
    return ret


def add_read_data_route(
    app: fastapi.APIRouter | fastapi.FastAPI,
    location: FeatureLocation,
    store: ContractStore,
) -> None:
    if location.location_type == "model":
        model_store = store.model(location.name)
        model = model_store.model
        view = model.predictions_view.as_view(model.name)
    else:
        view = store.feature_view(location.name).view

    view_request = view.retrieval_request
    entities = view_request.entities

    properties = {
        entity.name: schema_for_dtype(
            entity.dtype, entity.description, entity.constraints
        )
        for entity in entities
    }
    fetch_entities_api_schema = {
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "required": [entity.name for entity in entities],
                        "type": "object",
                        "properties": {
                            name: {"type": "array", "items": prop}
                            for name, prop in properties.items()
                        },
                        "additionalProperties": False,
                    }
                }
            },
            "required": True,
        },
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                feat.name: {
                                    "type": "array",
                                    "items": schema_for_dtype(
                                        feat.dtype, feat.description, feat.constraints
                                    ),
                                }
                                for feat in view_request.all_returned_features
                            },
                            "additionalProperties": False,
                        }
                    }
                }
            }
        },
    }

    fetch_entity_api_schema = {
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "required": [entity.name for entity in entities],
                        "type": "object",
                        "properties": properties,
                        "additionalProperties": False,
                    }
                }
            },
            "required": True,
        },
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                feat.name: schema_for_dtype(
                                    feat.dtype, feat.description, feat.constraints
                                )
                                for feat in view_request.all_returned_features
                            },
                            "additionalProperties": False,
                        }
                    }
                }
            }
        },
    }

    route_name = location.name.replace("_", "-")

    @app.post(
        f"/contracts/{route_name}/read/entities",
        openapi_extra=fetch_entities_api_schema,
        tags=[location.name],
    )
    async def read_entities(entities: dict[str, list[Any]]) -> list[dict[str, Any]]:
        start_time = monotonic()
        if location.location_type == "model":
            df = await store.model(location.name).predictions_for(entities).to_polars()
        else:
            df = (
                await store.feature_view(location.name)
                .features_for(entities)
                .to_polars()
            )

        res = df.to_dicts()

        total_time = monotonic() - start_time
        logger.info(f"Total time used {total_time:.4}")

        return res

    @app.post(
        f"/contracts/{route_name}/read/entity",
        openapi_extra=fetch_entity_api_schema,
        tags=[location.name],
    )
    async def read_entity(entity: dict[str, Any]) -> dict:
        first_key = next(iter(entity.keys()))
        assert not isinstance(
            entity[first_key], list
        ), "Expects only one entity. Consider using the /entities request instead."

        start_time = monotonic()
        if location.location_type == "model":
            df = await store.model(location.name).predictions_for(entity).to_polars()
        else:
            df = (
                await store.feature_view(location.name).features_for(entity).to_polars()
            )

        assert df.height == 1, f"Expected only one value, got {df.height}."
        res = df.to_dicts()[0]

        total_time = monotonic() - start_time
        logger.info(f"Total time used {total_time:.4}")

        return res


def add_infer_route(
    route: fastapi.APIRouter | fastapi.FastAPI,
    location: FeatureLocation,
    store: ContractStore,
) -> None:
    assert location.location_type == "model"
    model = store.model(location.name)

    assert (
        model.has_exposed_model()
    ), f"Model '{location.name}' needs to have an exposed model to infer."

    route_name = location.name.replace("_", "-")

    view_request = model.model.predictions_view.as_view(location.name).retrieval_request
    entities = view_request.entities

    properties = {
        entity.name: schema_for_dtype(
            entity.dtype, entity.description, entity.constraints
        )
        for entity in entities
    }
    entities_api_schema = {
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "required": [entity.name for entity in entities],
                        "type": "object",
                        "properties": {
                            name: {"type": "array", "items": prop}
                            for name, prop in properties.items()
                        },
                        "additionalProperties": False,
                    }
                }
            },
            "required": True,
        },
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                feat.name: {
                                    "type": "array",
                                    "items": schema_for_dtype(
                                        feat.dtype, feat.description, feat.constraints
                                    ),
                                }
                                for feat in view_request.all_returned_features
                            },
                            "additionalProperties": False,
                        }
                    }
                }
            }
        },
    }

    @route.post(
        f"/contracts/{route_name}/infer/entities",
        openapi_extra=entities_api_schema,
        tags=[route_name],
    )
    async def infer(entities: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output = await model.predict_over(entities).to_polars()
        return output.to_dict(as_series=False)


@overload
def router_for_store(store: ContractStore) -> fastapi.APIRouter: ...


@overload
def router_for_store(
    store: ContractStore, expose_tag: str | None
) -> fastapi.APIRouter: ...


@overload
def router_for_store(
    store: ContractStore, expose_tag: str | None, app: fastapi.FastAPI | None
) -> None: ...


def router_for_store(
    store: ContractStore,
    expose_tag: str | None = None,
    app: fastapi.FastAPI | None = None,
) -> fastapi.APIRouter | None:
    """
    Creates a FastAPI router that exposes all contracts for a given tag.


    If a contract have a source will it setup.

    `/contracts/{contract-name}/read/entity`
    `/contracts/{contract-name}/read/entities`

    and if a model contract have an exposed model will it also setup

    `/contracts/{contract-name}/infer/entities`

    Args:
        store (ContractStore): The contract store to expose.
        expose_tag (str | None): the tag to expose. Will default to `is_exposed_through_api`.

    Returns:
        APIRouter: A fastapi router
    """
    route = app or fastapi.APIRouter(tags=["Contracts"])

    data_to_expose = []
    infer_models = []

    for view in store.feature_views.values():
        if expose_tag:
            if view.tags and expose_tag in view.tags:
                data_to_expose.append(FeatureLocation.feature_view(view.name))
        else:
            data_to_expose.append(FeatureLocation.feature_view(view.name))

    for model in store.models.values():
        should_add = False
        if expose_tag:
            if model.tags and expose_tag in model.tags:
                should_add = True
        else:
            should_add = True

        if should_add:
            loc = FeatureLocation.model(model.name)

            is_exposed = False

            if model.exposed_model is not None:
                infer_models.append(loc)
                is_exposed = True

            if model.predictions_view.source is not None:
                data_to_expose.append(loc)
                is_exposed = True

            if not is_exposed:
                logger.info(
                    f"Unable to expose model '{model.name}' as there is no exposed_model or output_source set."
                )

    for data_loc in data_to_expose:
        add_read_data_route(route, data_loc, store)

    for model_loc in infer_models:
        add_infer_route(route, model_loc, store)

    if app is None:
        return route  # type: ignore
