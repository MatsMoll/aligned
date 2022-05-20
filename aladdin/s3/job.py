from io import StringIO
from aladdin.job_factory import JobFactory
from aladdin.request.retrival_request import RetrivalRequest
from aladdin.retrival_job import FactualRetrivalJob
from aladdin.s3.config import AwsS3DataSource
from aladdin.s3.storage import AwsS3Storage
from aladdin.feature import Feature

import pandas as pd

class AwsS3FactualJob(FactualRetrivalJob):
    
    source: AwsS3DataSource
    requests: set[RetrivalRequest]
    facts: dict[str, list]

    @property
    def storage(self) -> AwsS3Storage:
        return AwsS3Storage(self.source.config)

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
            
            mask = pd.Series.repeat(pd.Series([True]), df.shape[0]).reset_index(drop=True)
            set_mask = pd.Series.repeat(pd.Series([True]), result.shape[0]).reset_index(drop=True)
            for entity in entity_names:
                entity_source_name = self.source.feature_identifier_for([entity])[0]
                mask = mask & (df[entity_source_name].isin(self.facts[entity]))

                set_mask = set_mask & (pd.Series(self.facts[entity]).isin(df[entity_source_name]))
            
            
            feature_df = df.loc[mask, request_features]
            result.loc[set_mask, all_names] = feature_df

        result = await self.ensure_types(result)
        result = await self.compute_derived_featuers(result)
        return result

    async def to_arrow(self) -> pd.DataFrame:
        return await super().to_arrow()


class AwsS3JobFactory(JobFactory):

    source = AwsS3DataSource

    def _facts(self, facts: dict[str, list], requests: dict[AwsS3DataSource, RetrivalRequest]) -> AwsS3FactualJob:
        if len(requests.keys()) != 1:
            raise ValueError(f"Only able to load one {self.source} at a time")

        data_source = list(requests.keys())[0]
        if not isinstance(data_source, self.source):
            raise ValueError(f"Only {self.source} is supported, recived: {data_source}")

        # Group based on config
        return AwsS3FactualJob(
            source=data_source,
            facts=facts,
            requests=list(requests.values())
        )