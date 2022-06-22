from datetime import datetime

from fastapi import FastAPI, HTTPException
from numpy import nan
from pydantic import BaseModel

from aladdin.feature import Feature
from aladdin.feature_source import WritableFeatureSource
from aladdin.feature_store import FeatureStore


class APIFeatureRequest(BaseModel):
    entities: dict[str, list]
    features: list[str]


class FastAPIServer:
    @staticmethod
    def feature_view_path(name: str, feature_store: FeatureStore, app: FastAPI, can_write: bool) -> None:

        feature_view = feature_store.feature_views[name]
        required_features: set[Feature] = feature_view.entities.union(feature_view.features)

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

        @app.post(f'/feature-view/{name}/all')
        async def all(limit: int | None = None) -> dict:
            df = await feature_store.feature_view(name).all(limit=limit).to_df()
            df.replace(nan, value=None, inplace=True)
            return df.to_dict('list')

        if can_write:

            @app.post(f'/feature-view/{name}/write', openapi_extra=write_api_schema)
            async def write(feature_values: dict) -> None:
                await feature_store.feature_view(name).write(feature_values)

    @staticmethod
    def model_path(name: str, feature_store: FeatureStore, app: FastAPI) -> None:
        feature_request = feature_store.model_requests[name]

        entities: set[Feature] = set()
        for request in feature_request.needed_requests:
            entities.update(request.entities)

        required_features = entities.copy()
        for request in feature_request.needed_requests:
            if isinstance(request, list):
                for sub_request in request:
                    required_features.update(sub_request.all_required_features)
            else:
                required_features.update(request.all_required_features)

        properties = {
            entity.name: {
                'type': 'array',
                'items': {'type': entity.dtype.name},
            }
            for entity in entities
        }
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
        # write_api_schema = {
        #     'requestBody': {
        #         'content': {
        #             'application/json': {
        #                 'schema': {
        #                     'required': [feature.name for feature in required_features],
        #                     'type': 'object',
        #                     'properties': {
        #                         feature.name: {
        #                             'type': 'array',
        #                             'items': {'type': feature.dtype.name},
        #                         }
        #                         for feature in required_features
        #                     },
        #                 }
        #             }
        #         },
        #         'required': True,
        #     },
        # }

        # Using POST as this can have a body with the fact / entity table
        @app.post(f'/model/{name}', openapi_extra=featch_api_schema)
        async def get_model(entity_values: dict) -> dict:
            missing_entities = {entity.name for entity in entities if entity.name not in entity_values}
            if missing_entities:
                raise HTTPException(status_code=400, detail=f'Missing entity values {missing_entities}')

            entity_values['event_timestamp'] = [
                datetime.fromtimestamp(value)
                if isinstance(value, (float, int))
                else datetime.fromisoformat(value)
                for value in entity_values['event_timestamp']
            ]

            df = await feature_store.model(name).features_for(entity_values).to_df()
            df.replace(nan, value=None, inplace=True)
            return df.to_dict('list')

        # @app.post(f'/model/{name}/write', openapi_extra=write_api_schema)
        # async def write_model(feature_values: dict) -> dict:
        #     missing_features = {
        #         entity.name for entity in required_features if entity.name not in feature_values
        #     }
        #     if missing_features:
        #         raise HTTPException(status_code=400, detail=f'Missing feature values {missing_features}')

        #     await feature_store.model(name).write(feature_values)

    @staticmethod
    def run(
        feature_store: FeatureStore,
        host: str | None = None,
        port: int | None = None,
        workers: int | None = None,
    ) -> None:
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI()
        app.docs_url = '/docs'

        for model in feature_store.all_models:
            FastAPIServer.model_path(model, feature_store, app)

        can_write_to_store = isinstance(feature_store.feature_source, WritableFeatureSource)

        for feature_view in feature_store.feature_views.keys():
            FastAPIServer.feature_view_path(feature_view, feature_store, app, can_write_to_store)

        @app.post('/features')
        async def features(payload: APIFeatureRequest) -> dict:
            df = await feature_store.features_for(
                payload.entities,
                features=payload.features,
            ).to_df()
            df.replace(nan, value=None, inplace=True)
            return df.to_dict('list')

        uvicorn.run(app, host=host or '127.0.0.1', port=port or 8000, workers=workers or workers)
