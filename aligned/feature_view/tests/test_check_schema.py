import pytest
from aligned import Bool, Float, String, feature_view, FileSource
from aligned.feature_view.feature_view import check_schema
from typing import Annotated
import pandas as pd


@feature_view(
    name='test',
    source=FileSource.parquet_at('test.parquet'),
)
class TestView:

    id = String().as_entity()

    a = String()
    b = Bool()
    c = Float()


@check_schema()
def some_method(df: Annotated[pd.DataFrame, TestView]) -> pd.DataFrame:
    return df


def test_check_schema() -> None:

    df = pd.DataFrame(
        {'id': ['a', 'b', 'c'], 'a': ['a', 'b', 'c'], 'b': [True, False, True], 'c': [1.0, 2.0, 3.0]}
    )

    res = some_method(df)

    assert df.equals(res)


def test_check_schema_error() -> None:

    df = pd.DataFrame(
        {
            'id': ['a', 'b', 'c'],
            'a': ['a', 'b', 'c'],
            'b': [True, False, True],
        }
    )

    with pytest.raises(ValueError):  # noqa: PT011
        some_method(df)
