import pytest
import polars as pl
from aligned.feature_view.feature_view import FeatureViewWrapper
from aligned import FileSource, feature_view, String, Int32


@feature_view(
    source=FileSource.parquet_at(""),
)
class Races:
    rittID = Int32().as_entity()
    year = Int32().as_entity()

    race_type = (
        String()
        .accepted_values(["T", "K"])
        .description("K for classics, T for tour - I think")
    )

    race_profile_image = (
        String()
        .transformed_using_features_polars(
            [rittID, year, race_type],
            pl.when(pl.col("race_type").str.to_uppercase() == "K")
            .then(
                pl.concat_str(
                    pl.lit("https://firstcycling.com/img/rittaar/"),
                    pl.col("rittID"),
                    pl.lit("_"),
                    pl.col("year"),
                    pl.lit(".jpg"),
                )
            )
            .otherwise(pl.lit(None)),
        )
        .as_image_url()
        .description("This image url only makes sense if the race is not a stage race")
    )
    race_profile_image_data = race_profile_image.load_bytes()


def test_intermediat_features_is_not_in_result():
    store = Races.query().store
    req = store.requests_for_features(
        [Races().race_profile_image_data.feature_reference()]
    )
    assert "race_profile_image" not in req.features_to_include


@pytest.mark.asyncio
async def test_fetch_all_request(titanic_feature_view: FeatureViewWrapper) -> None:
    compiled_view = titanic_feature_view.compile()
    request = compiled_view.request_all

    expected_features = {
        "age",
        "name",
        "sex",
        "survived",
        "sibsp",
        "cabin",
        "has_siblings",
        "is_male",
        "is_female",
        "is_mr",
    }

    assert not request.needs_event_timestamp
    assert len(request.needed_requests) == 1

    retrieval_request = request.needed_requests[0]
    missing_features = expected_features - retrieval_request.all_feature_names
    assert (
        retrieval_request.all_feature_names == expected_features
    ), f"Missing features {missing_features}"
    assert retrieval_request.entity_names == {"passenger_id"}

    assert len(request.request_result.entities) == 1
    assert len(request.request_result.features) == len(expected_features)


@pytest.mark.asyncio
async def test_fetch_features_request(titanic_feature_view: FeatureViewWrapper) -> None:
    compiled_view = titanic_feature_view.compile()
    wanted_features = {"cabin", "is_male"}
    request = compiled_view.request_for(wanted_features)
    expected_features = {"sex", "cabin", "is_male"}
    assert not request.needs_event_timestamp
    assert len(request.needed_requests) == 1

    retrieval_request = request.needed_requests[0]
    missing_features = expected_features - retrieval_request.all_feature_names
    # All the features to retrieval and computed
    assert (
        retrieval_request.all_feature_names == expected_features
    ), f"Missing features {missing_features}"
    assert retrieval_request.entity_names == {"passenger_id"}

    # All the features that is returned
    assert len(request.request_result.entities) == 1
    assert len(request.request_result.features) == len(wanted_features)
