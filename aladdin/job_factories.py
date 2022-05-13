from aladdin.job_factory import JobFactory


custom_factories: set[JobFactory] = set()

def get_factories() -> set[JobFactory]:
    from aladdin.psql.data_source import PostgreSQLRetrivalJob
    return {
        PostgreSQLRetrivalJob()
    }.union(custom_factories)