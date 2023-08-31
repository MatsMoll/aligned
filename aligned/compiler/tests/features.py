from aligned import feature_view, UUID, String, PostgreSQLConfig, Float

source = PostgreSQLConfig.localhost('test')


@feature_view(name='test', description='test', batch_source=source.table('test'))
class Test:
    id = UUID().as_entity()

    x = String()
    y = Float()

    z = y**2
