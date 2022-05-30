from aladdin.job_factory import JobFactory


custom_factories: set[JobFactory] = set()

def get_factories() -> set[JobFactory]:
    from aladdin.psql.factory import PostgresJobFactory
    from aladdin.s3.job import AwsS3JobFactory
    from aladdin.local.factory import LocalFileJobFactory
    from aladdin.redshift.factory import RedshiftJobFactory
    factories = {
        PostgresJobFactory(),
        AwsS3JobFactory(), 
        LocalFileJobFactory(),
        RedshiftJobFactory(),
    }.union(custom_factories)
    return {factory.source.type_name: factory for factory in factories}