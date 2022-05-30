from dataclasses import dataclass
from io import StringIO

from sqlalchemy import all_
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import FactualRetrivalJob, FullExtractJob
from aladdin.s3.storage import FileStorage, HttpStorage
from aladdin.feature import Feature
from aladdin.local.source import FileSource

import pandas as pd

from aladdin.storage import Storage

@dataclass
class LocalFileFullJob(FullExtractJob):

    source: FileSource
    request: RetrivalRequest
    limit: int | None

    @property
    def storage(self) -> Storage:
        if self.source.path.startswith("http"):
            return HttpStorage()
        else:
            return FileStorage()

    async def _to_df(self) -> pd.DataFrame:
        content = await self.storage.read(self.source.path)
        file = StringIO(str(content, "utf-8"))
        entity_names = self.request.entity_names
        all_names = list(self.request.all_required_feature_names.union(entity_names))
        request_features = self.source.feature_identifier_for(all_names)
        df = pd.read_csv(file)
        df.rename(columns={org_name: wanted_name for org_name, wanted_name in  zip(request_features, all_names)}, inplace=True)

        if self.limit and df.shape[0] > self.limit:
            return df.iloc[:self.limit]
        else:
            return df

    async def to_arrow(self) -> pd.DataFrame:
        return await super().to_arrow()

@dataclass
class LocalFileFactualJob(FactualRetrivalJob):
    
    source: FileSource
    requests: set[RetrivalRequest]
    facts: dict[str, list]

    @property
    def storage(self) -> Storage:
        if self.source.path.startswith("http"):
            return HttpStorage()
        else:
            return FileStorage()

    async def _to_df(self) -> pd.DataFrame:
        content = await self.storage.read(self.source.path)
        file = StringIO(str(content, "utf-8"))
        df = pd.read_csv(file)
        all_features: set[Feature] = set()
        for request in self.requests:
            all_features.update(request.all_required_features)

        number_of_facts = len(list(self.facts.values())[0])
        result = pd.DataFrame(index=range(number_of_facts))

        for request in self.requests:
            entity_names = request.entity_names
            all_names = request.all_required_feature_names.union(entity_names)
            request_features = self.source.feature_identifier_for(all_names)

            print(df[self.source.feature_identifier_for(entity_names)])
            
            mask = pd.Series.repeat(pd.Series([True]), df.shape[0]).reset_index(drop=True)
            set_mask = pd.Series.repeat(pd.Series([True]), result.shape[0]).reset_index(drop=True)
            for entity in entity_names:
                entity_source_name = self.source.feature_identifier_for([entity])[0]
                mask = mask & (df[entity_source_name].isin(self.facts[entity]))

                set_mask = set_mask & (pd.Series(self.facts[entity]).isin(df[entity_source_name]))

            print(mask.sum())
            print(set_mask.sum())
            
            feature_df = df.loc[mask, request_features]
            feature_df.rename(columns={org_name: wanted_name for org_name, wanted_name in  zip(request_features, all_names)}, inplace=True)
            result.loc[set_mask, list(all_names)] = feature_df.reset_index(drop=True)

        print(result)
        return result

    async def to_arrow(self) -> pd.DataFrame:
        return await super().to_arrow()
