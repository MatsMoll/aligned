from collections import defaultdict
from dataclasses import dataclass
from aladdin.data_source.batch_data_source import BatchDataSource
from aladdin.feature_source import BatchFeatureSource, FeatureSource
from aladdin.feature_view.feature_view import FeatureView
from aladdin.online_source import OnlineSource
from aladdin.feature_view.combined_view import CombinedFeatureView, CompiledCombinedFeatureView

from aladdin.feature_view.compiled_feature_view import CompiledFeatureView
from aladdin.repo_definition import RepoDefinition
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import RetrivalJob

@dataclass
class RawStringFeatureRequest:
    features: set[str]

    @property
    def feature_view_names(self) -> set[str]:
        return {RawStringFeatureRequest.unpack_feature(feature)[0] for feature in self.features}

    @property
    def grouped_features(self) -> dict[str, set[str]]:
        unpacked_features = [RawStringFeatureRequest.unpack_feature(feature) for feature in self.features]
        grouped = defaultdict(set)
        for feature_view, feature in unpacked_features:
            grouped[feature_view].add(feature)
        return grouped

    @staticmethod
    def unpack_feature(feature: str) -> tuple[str, str]:
        splits = feature.split(":")
        if len(splits) != 2:
            raise ValueError(f"Invalid feature name: {feature}")
        return (splits[0], splits[1])


class FeatureStore:

    feature_source: FeatureSource
    feature_views: dict[str, CompiledFeatureView]
    combined_feature_views: dict[str, CompiledCombinedFeatureView]
    model_requests: dict[str, list[RetrivalRequest]]

    @property
    def all_models(self) -> list[str]:
        return list(self.model_requests.keys())


    def __init__(self, feature_views: set[CompiledFeatureView], combined_feature_views: set[CompiledCombinedFeatureView], models: dict[str, list[str]], feature_source: FeatureSource) -> None:
        self.feature_source = feature_source
        self.combined_feature_views = {fv.name: fv for fv in combined_feature_views}
        self.feature_views = {fv.name: fv for fv in feature_views}
        self.model_requests = {name: self.requests_for(RawStringFeatureRequest(model)) for name, model in models.items()}


    @staticmethod
    def from_definition(repo: RepoDefinition, feature_source: FeatureSource | None = None) -> "FeatureStore":
        source = feature_source or repo.online_source.feature_source(repo.feature_views)
        return FeatureStore(
            feature_views=repo.feature_views,
            combined_feature_views=repo.combined_feature_views,
            models=repo.models,
            feature_source=source
        )


    def features_for(self, facts: dict[str, list], features: list[str]) -> RetrivalJob:
        feature_request = RawStringFeatureRequest(features=set(features))
        requests = self.requests_for(feature_request)
        feature_names = set()
        for feature_set in feature_request.grouped_features.values():
            feature_names.update(feature_set)
        for request in requests:
            feature_names.update(request.entity_names)
        return self.feature_source.features_for(facts, requests, feature_names)

    def model(self, name: str) -> "OfflineModelStore":
        return OfflineModelStore(
            self.feature_source,
            self.model_requests[name]
        )

    def requests_for(self, feature_request: RawStringFeatureRequest) -> list[RetrivalRequest]:
        features = feature_request.grouped_features
        requests: list[RetrivalRequest] = []
        for feature_view_name in feature_request.feature_view_names:
            if feature_view_name in self.combined_feature_views:
                cfv = self.combined_feature_views[feature_view_name]
                requests.extend(
                    cfv.requests_for(features[feature_view_name])
                )
            else:
                feature_view = self.feature_views[feature_view_name]
                requests.append(
                    feature_view.request_for(features[feature_view_name])
                )
        return requests

    def add_feature_view(self, feature_view: FeatureView):
        compiled_view = type(feature_view).compile()
        self.feature_views[compiled_view.name] = compiled_view
        if isinstance(self.feature_source, BatchFeatureSource):
            self.feature_source.sources[compiled_view.name] = compiled_view.batch_data_source

    def add_combined_feature_view(self, feature_view: CombinedFeatureView):
        compiled_view = type(feature_view).compile()
        self.combined_feature_views[compiled_view.name] = compiled_view

    def all_for(self, view: str, limit: int | None = None) -> RetrivalJob:
        return self.feature_source.all_for(
            self.feature_views[view].request_all,
            limit
        )

@dataclass
class OfflineModelStore:

    source: FeatureSource
    requests: set[RetrivalRequest]

    def features_for(self, facts: dict[str, list]) -> RetrivalJob:

        feature_names = set()
        for request in self.requests:
            feature_names.update(request.all_feature_names)
            feature_names.update(request.entity_names)
            
        return self.source.features_for(facts, self.requests, feature_names)

    async def write(self, values: dict[str]):
        pass


"""
Different stores and use cases:

DS personel, which will fetch data sets for testing features, feature enginering and testing models

Wanted behavior:
- Join data x and see if it could be interesting for prediction y
- Fetch data set in timeframe ... for partition ...
- Store data sets for sharing etc.

DE personel, which wants to setup pipeline for training, and serving models:

Wanted behavior:
- Setup a DAG, for resilient pipeliens
- Monitor pipelines
- Serve features in a fast way


Online Serving:
Needs to have the freshest data for model x and entities ...
Only reason for a connection to the batch sources is to load "uncached" featues
If using the push api, would a local processing / spark cluster access be needed.
Should offload the processing to a worker tho.


Offline Serving:
No need for "freshest" features, therfore no connection to online.
Only need to have a connection to the batch, and local processing. May need to connect to a local spark cluster, maybe local Airflow if wanted?
May need to have access to a regristry like S3 for artifacts


Pipeline processing:
Needs to have a trigger to start on event x
Needs a connection to the batch and online sources. As it will update the online store.
May need to connect to clusters here and there. E.g: Spark cluster, + Airflow container, databases etc.
May need to have access to a regristry like S3 for artifacts



Some potential solutions for the problem

All solutions will be based on sending the RepoDefinition data model as a parameter

Current solution:
Use a factory pattern where each factory is associated with an identifier. E.g: "psql" -> PostgresJobFactory
This job will thereafter have the possibility to define which format the user wants it in.
This makes it possible to support DAG's where they can choose the format.

By having a "RetrivalJob" class, can we also add data validation, and sentralize the transformation logic.
However, it could get hard to fit all edge cases.

If we would add DAG functionality, would the following be needed.
- Generate the different opperations for the whole feature store.
    This would mean, extracting the data as one method (Run the first step of the retrival job).
    Another could be ensuring data types, and converting them into the expected.
        Maybe also to data checks on "core" features.
    Thired could be running stateless transformations.
    Forth could be running CombinedFeatureView transformations.
        This would potentially need to order the transformations if a combined view depends on another combined view.
    (?) Potentially run data checks (Great Expectations) on data here
    Fifth: Load the data. This would most likely be into the online store.
        If it would have been running localy (aka. for data science), could we load the data into a dataset store, local dir, or in RAM.


Problem here:
All steps may use different formats for each step. It would therefore be nice to make it possible to share the same data format across each step.
Now this may not be possible.
One potential solution is to use some predifined storage foramt. e.g: Parquet between each step given that it is different containers / processes doing each step.
Another solution could be to define a restriction of formats between certan steps. E.g: ensuring data types -> combined view transformations is either pandas, or spark dataframe.
The E and L part is different tho. However by having n -> m formats. E.g: dict, csv, arrow, parquest, sql -> pandas, spark. May make it easier.
Problem with spark is that it needs to load the file, because of e.g: HDFS. Therefore it needs it's own extract and load logic.


Could create a system that defines the input, output for each subtask, and then link all them up.
I think this is what dagster does.
"""
