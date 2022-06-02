from aladdin.job_factory import JobFactory

custom_factories: set[JobFactory] = set()


def get_factories() -> dict[str, JobFactory]:
    from aladdin.local.factory import LocalFileJobFactory
    from aladdin.psql.factory import PostgresJobFactory
    from aladdin.redshift.factory import RedshiftJobFactory
    from aladdin.s3.factory import AwsS3JobFactory

    factories = {
        PostgresJobFactory(),
        AwsS3JobFactory(),
        LocalFileJobFactory(),
        RedshiftJobFactory(),
    }.union(custom_factories)
    return {factory.source.type_name: factory for factory in factories}
