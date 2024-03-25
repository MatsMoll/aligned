from aligned import Directory, AwsS3Config, FileSource, ParquetConfig, CsvConfig


def test_directory_interfaces() -> None:
    aws_config = AwsS3Config('', '', '', '')

    dirs = [aws_config, FileSource]

    parquet_config = ParquetConfig(compression='snappy')
    csv_config = CsvConfig(seperator=',')
    mapping_keys = {'key': 'value'}

    for config in dirs:

        directory: Directory = config.directory('path')

        sub_dir = directory.sub_directory('sub_path')

        parquet = sub_dir.parquet_at('test.parquet', mapping_keys=mapping_keys, config=parquet_config)
        csv = sub_dir.csv_at('test.csv', mapping_keys=mapping_keys, csv_config=csv_config)
        json = sub_dir.json_at('test.json')

        assert parquet is not None
        assert csv is not None
        assert json is not None
