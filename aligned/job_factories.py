from aligned.job_factory import JobFactory

custom_factories: set[JobFactory] = set()


def get_factories() -> dict[str, JobFactory]:
    from aligned.local.factory import CsvFileJobFactory, ParquetFileJobFactory
    from aligned.psql.factory import PostgresJobFactory
    from aligned.redshift.factory import RedshiftJobFactory
    from aligned.s3.factory import AwsS3JobFactory

    factories: set[JobFactory] = {
        PostgresJobFactory(),
        CsvFileJobFactory(),
        ParquetFileJobFactory(),
        RedshiftJobFactory(),
        AwsS3JobFactory(),
    }.union(custom_factories)

    return {factory.source.type_name: factory for factory in factories}
