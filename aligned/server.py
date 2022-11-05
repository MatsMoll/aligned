import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from numpy import nan
from pydantic import BaseModel

from aligned.data_source.stream_data_source import HttpStreamSource
from aligned.feature_source import WritableFeatureSource
from aligned.feature_store import FeatureStore
from aligned.schemas.feature import Feature
from aligned.schemas.feature_view import CompiledFeatureView

logger = logging.getLogger(__name__)


class APIFeatureRequest(BaseModel):
    entities: dict[str, list]
    features: list[str]


@dataclass
class TopicInfo:
    name: str
    views: list[CompiledFeatureView]
    mappings: dict[str, str]


class FastAPIServer:
    @staticmethod
    def write_to_topic_path(topic: TopicInfo, feature_store: FeatureStore, app: FastAPI) -> None:

        required_features: set[Feature] = set()
        for view in topic.views:
            required_features.update(view.entities.union(view.features))

        view_names = [view.name for view in topic.views]
        mappings: dict[str, list[str]] = {
            output: input_path.split('.') for input_path, output in topic.mappings.items()
        }

        write_api_schema = {
            'requestBody': {
                'content': {
                    'application/json': {
                        'schema': {
                            'required': [feature.name for feature in required_features],
                            'type': 'object',
                            'properties': {
                                feature.name: {
                                    'type': 'array',
                                    'items': {'type': feature.dtype.name},
                                }
                                for feature in required_features
                            },
                        }
                    }
                },
                'required': True,
            },
        }

        @app.post(f'/topics/{topic.name}/write', openapi_extra=write_api_schema)
        async def write(feature_values: dict) -> None:

            for output_name, input_path in mappings.items():

                if output_name in feature_values:
                    continue

                if len(input_path) == 1:
                    feature_values[output_name] = feature_values[input_path[0]]
                else:
                    from functools import reduce

                    def find_path_variable(values: dict, key: str) -> Any:
                        return values.get(key, {})

                    values = [
                        # Select the value based on the key path given
                        reduce(find_path_variable, input_path[1:], value)  # initial value
                        for value in feature_values[input_path[0]]
                    ]
                    feature_values[output_name] = [value if value != {} else None for value in values]

            await asyncio.gather(
                *[feature_store.feature_view(view_name).write(feature_values) for view_name in view_names]
            )

    @staticmethod
    def feature_view_path(name: str, feature_store: FeatureStore, app: FastAPI) -> None:
        @app.post(f'/feature-views/{name}/all')
        async def all(limit: int | None = None) -> dict:
            df = await feature_store.feature_view(name).all(limit=limit).to_df()
            df.replace(nan, value=None, inplace=True)
            return df.to_dict('list')

    @staticmethod
    def model_path(name: str, feature_store: FeatureStore, app: FastAPI) -> None:
        feature_request = feature_store.model_requests[name]

        entities: set[Feature] = set()
        for request in feature_request.needed_requests:
            entities.update(request.entities)

        required_features = entities.copy()
        for request in feature_request.needed_requests:
            required_features.update(request.all_required_features)

        properties = {
            entity.name: {
                'type': 'array',
                'items': {'type': entity.dtype.name},
            }
            for entity in entities
        }
        needs_event_timestamp = feature_request.needs_event_timestamp
        if needs_event_timestamp:
            properties['event_timestamp'] = {'type': 'array', 'items': {'type': 'string'}}

        featch_api_schema = {
            'requestBody': {
                'content': {
                    'application/json': {
                        'schema': {
                            'required': [entity.name for entity in entities] + ['event_timestamp'],
                            'type': 'object',
                            'properties': properties,
                        }
                    }
                },
                'required': True,
            },
        }

        # Using POST as this can have a body with the fact / entity table
        @app.post(f'/models/{name}', openapi_extra=featch_api_schema)
        async def get_model(entity_values: dict) -> str:
            missing_entities = {entity.name for entity in entities if entity.name not in entity_values}
            if missing_entities:
                raise HTTPException(status_code=400, detail=f'Missing entity values {missing_entities}')

            if needs_event_timestamp:
                entity_values['event_timestamp'] = [
                    datetime.fromtimestamp(value)
                    if isinstance(value, (float, int))
                    else datetime.fromisoformat(value)
                    for value in entity_values['event_timestamp']
                ]

            df = await feature_store.model(name).features_for(entity_values).to_df()
            orient = 'values'
            body = ','.join([f'"{column}":{df[column].to_json(orient=orient)}' for column in df.columns])
            return Response(content=f'{{{body}}}', media_type='application/json')

    @staticmethod
    def app(feature_store: FeatureStore) -> FastAPI:
        from asgi_correlation_id import CorrelationIdMiddleware
        from fastapi import FastAPI
        from fastapi.middleware import Middleware

        app = FastAPI(middleware=[Middleware(CorrelationIdMiddleware)])
        app.docs_url = '/docs'

        for model in feature_store.all_models:
            FastAPIServer.model_path(model, feature_store, app)

        can_write_to_store = isinstance(feature_store.feature_source, WritableFeatureSource)

        topics: dict[str, TopicInfo] = {}

        for feature_view in feature_store.feature_views.values():
            if not (stream_source := feature_view.stream_data_source):
                continue

            if isinstance(stream_source, HttpStreamSource):
                topic_name = stream_source.topic_name
                if topic_name not in topics:
                    topics[topic_name] = TopicInfo(
                        name=topic_name, views=[feature_view], mappings=stream_source.mappings
                    )
                else:
                    info = topics[topic_name]

                    topics[topic_name] = TopicInfo(
                        name=topic_name,
                        views=info.views + [feature_view],
                        mappings=info.mappings | stream_source.mappings,
                    )

        if can_write_to_store:
            for topic in topics.values():
                FastAPIServer.write_to_topic_path(topic, feature_store, app)
        else:
            logger.info(
                (
                    'Warning! The server is not using a WritableFeatureSource, ',
                    'and can therefore not setup stream sources',
                )
            )

        @app.post('/features')
        async def features(payload: APIFeatureRequest) -> dict:
            df = await feature_store.features_for(
                payload.entities,
                features=payload.features,
            ).to_df()
            orient = 'values'
            body = ','.join([f'"{column}":{df[column].to_json(orient=orient)}' for column in df.columns])
            return Response(content=f'{{{body}}}', media_type='application/json')

        return app

    @staticmethod
    def run(
        feature_store: FeatureStore,
        host: str | None = None,
        port: int | None = None,
        workers: int | None = None,
    ) -> None:
        import uvicorn

        app = FastAPIServer.app(feature_store)

        uvicorn.run(app, host=host or '127.0.0.1', port=port or 8000, workers=workers or workers)
