# Aligned

A data managment tool for ML applications.

Similar to have DBT is a data managment tool for business analytics, will Aligned manage ML projects.

Aligned does this through two things.
1. A light weight data managment system. Making it possible to query a data lake and databases.
2. Tooling to define a `model_contract`. Clearing up common unanswerd questions through code.


Furthermore, Aligned collect data lineage between models, basic feature transformations. While also making it easy to reduce data leakage with point-in-time valid data and fix other problems described in [Sculley et al. [2015]](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf).

## Examples

Bellow are some examples of how Aligned can be used.

### Aligned UI

Aligned provides an UI to view which data exists, the expectations we have and find faults.

[View the example UI](https://aligned-catalog.azurewebsites.net/).
However, this is still under development, so sign up for a [wait list](https://aligned-managed-web.vercel.app/) to get access.


### Example Repo

Want to look at examples of how to use `aligned`?
View the [`MatsMoll/aligned-example` repo](https://github.com/MatsMoll/aligned-example).

Or see how you could query a file in a data lake.

```python
store = await FeatureStore.from_dir(".")
df = await store.execute_sql("SELECT * FROM titanic LIMIT 10").to_polars()
```

## Docs

Check out the [Aligned Docs](https://www.aligned.codes), but keep in mind that they are still work in progress.

---

### Available Features

Bellow are some of the features Aligned offers:

- [Data Catalog](https://aligned-managed-web.vercel.app/)
- [Data Lineage](https://aligned-managed-web.vercel.app/)
- [Model Performance Monitoring](https://aligned-managed-web.vercel.app/)
- [Data Freshness](#data-freshness)
- [Data Quality Assurance](#data-quality)
- [Easy Data Loading](#access-data)
- [Feature Store](https://matsmoll.github.io/posts/understanding-the-chaotic-landscape-of-mlops#feature-store)
- [Load Form Multiple Sources](#fast-development)
- [Feature Server](#feature-server)
- [Stream Processing](#stream-worker)


All from the simple API of defining
- [Data Sources](#data-sources)
- [Feature Views](#feature-views)
- [Models](#describe-models)

As a result, loading model features is as easy as:

```python
entities = {"passenger_id": [1, 2, 3, 4]}
await store.model("titanic").features_for(entities).to_pandas()
```

Aligned is still in active development, so changes are likely.

## Model Contract

Aligned introduces a new concept called the "model contract", which tries to answer the following questions.

- What is predicted?
- What is assosiated with a prediction? - A user id?
- Where do we store predictions?
- Do a model depend on other models?
- Is the model exposed through an API?
- What needs to be sent in, to use the model?
- Is it classification, regression, gen ai?
- Where is the ground truth stored? - if any
- Who owns the model?
- Where do we store data sets?

All this is described through a `model_contract`, as shown bellow.

```python
@model_contract(
    name="eta_taxi",
    features=[
        trips.eucledian_distance,
        trips.number_of_passengers,
        traffic.expected_delay
    ],
    prediction_source=FileSource.delta_at("titanic_model/predictions")
)
class EtaTaxi:
    trip_id = Int32().as_entity()
    predicted_at = EventTimestamp()
    predicted_duration = trips.duration.as_regression_target()
```

## Data Sources

Alinged makes handling data sources easy, as you do not have to think about how it is done.

Furthermore, Aligned makes it easy to switch parts of the business logic to a local setup for debugging purposes.

```python
from aligned import FileSource, AwsS3Config, AzureBlobConfig, Directory
import os

root_directory: Directory = FileSource.directory("my-awesome-project")

if os.getenv("USE_AWS", "false").lower() == "true":

    aws_config = AwsS3Config(...)
    root_directory = aws_config.directory("my-awesome-project")

elif os.getenv("USE_AZURE", "false").lower() == "true":

    azure_config = AzureBlobConfig(...)
    root_directory = azure_config.directory("my-awesome-project")


taxi_project = root_directory.sub_directory("eta_taxi")

csv_source = taxi_project.csv_at("predictions.csv")
parquet_source = taxi_project.parquet_at("predictions.parquet")
delta_source = taxi_project.delta_at("predictions")
```

### Date Formatting
Managing a data lake can be hard. However, a common problem when using file formats can be managing date formats. As a result do Aligned provide a way to standardise this, so you can focus on what matters.

```python
from aligned import FileSource
from aligned.schemas.date_formatter import DateFormatter

iso_formatter = DateFormatter.iso_8601()
unix_formatter = DateFormatter.uniq_timestamp(time_unit="us", time_zone="UTC")
custom_strtime_formatter = DateFormatter.string_format("%Y/%m/%d %H:%M:%S")

FileSource.csv_at("my/file.csv", date_formatter=unix_formatter)
```

## Feature Views

Aligned also makes it possible to define data and features through `feature_view`s.
Then get code completion and typesafety by referencing them in other features.

This makes the features light weight, data source independent, and flexible.

```python
@feature_view(
    name="passenger",
    description="Some features from the titanic dataset",
    source=FileSource.csv_at("titanic.csv"),
    materialized_source=FileSource.parquet_at("titanic.parquet"),
)
class TitanicPassenger:

    passenger_id = Int32().as_entity()

    age = (
        Float()
            .description("A float as some have decimals")
            .lower_bound(0)
            .upper_bound(110)
    )

    name = String()
    sex = String().accepted_values(["male", "female"])
    did_survive = Bool().description("If the passenger survived")
    sibsp = Int32().lower_bound(0).description("Number of siblings on titanic")
    cabin = String().is_optional()

    # Creates two one hot encoded values
    is_male, is_female = sex.one_hot_encode(['male', 'female'])
```

### Fast development

Making iterativ and fast exploration in ML is important. This is why Aligned also makes it super easy to combine, and test multiple sources.

```python
my_db = PostgreSQLConfig.localhost()

aws_bucket = AwsS3Config(...)

@feature_view(
    name="passengers",
    description="...",
    source=my_db.table("passengers")
)
class TitanicPassenger:

    passenger_id = Int32().as_entity()

    # Some features
    ...

# Change data source
passenger_view = TitanicPassenger.query()

psql_passengers = await passenger_view.all().to_pandas()
aws_passengers = await passenger_view.using_source(
    aws_bucket.parquet_at("passengers.parquet")
).to_pandas()

```

## Describe Models

Usually will you need to combine multiple features for each model.
This is where a `Model` comes in.
Here can you define which features should be exposed.

```python
passenger = TitanicPassenger()
location = LocationFeatures()

@model_contract(
    name="titanic",
    features=[ # aka. the model input
        passenger.constant_filled_age,
        passenger.ordinal_sex,
        passenger.sibsp,

        location.distance_to_shore,
        location.distance_to_closest_boat
    ]
)
class Titanic:

    # Referencing the passenger's survived feature as the target
    did_survive = passenger.survived.as_classification_target()
```

## Data Freshness
Making sure a source contains fresh data is a crucial part to create propper ML applications.
Therefore, Aligned provides an easy way to check how fresh a source is.

```python
@feature_view(
    name="departures",
    description="Features related to the departure of a taxi ride",
    source=taxi_db.table("departures"),
)
class TaxiDepartures:

    trip_id = UUID().as_entity()

    pickuped_at = EventTimestamp()

    number_of_passengers = Int32()

    dropoff_latitude = Float().is_required()
    dropoff_longitude = Float().is_required()

    pickup_latitude = Float().is_required()
    pickup_longitude = Float().is_required()


freshness = await TaxiDepartures.freshness_in_batch_source()

if freshness < datetime.now() - timedelta(days=2):
    raise ValueError("To old data to create an ML model")
```

## Access Data

You can easily create a feature store that contains all your feature definitions.
This can then be used to genreate data sets, setup an instce to serve features, DAG's etc.

```python
store = await FileSource.json_at("./feature-store.json").feature_store()

# Select all features from a single feature view
df = await store.all_for("passenger", limit=100).to_pandas()
```

### Centraliced Feature Store Definition
You would often share the features with other coworkers, or split them into different stages, like `staging`, `shadow`, or `production`.
One option is therefore to reference the storage you use, and load the `FeatureStore` from there.

```python
aws_bucket = AwsS3Config(...)
store = await aws_bucket.json_at("production.json").feature_store()

# This switches from the production online store to the offline store
# Aka. the batch sources defined on the feature views
experimental_store = store.offline_store()
```
This json file can be generated by running `aligned apply`.

### Select multiple feature views

```python
df = await store.features_for({
    "passenger_id": [1, 50, 110]
}, features=[
    "passenger:scaled_age",
    "passenger:is_male",
    "passenger:sibsp"

    "other_features:distance_to_closest_boat",
]).to_polars()
```

### Model Service

Selecting features for a model is super simple.


```python
df = await store.model("titanic_model").features_for({
    "passenger_id": [1, 50, 110]
}).to_pandas()
```

### Feature View

If you want to only select features for a specific feature view, then this is also possible.

```python
prev_30_days = await store.feature_view("match").previous(days=30).to_pandas()
sample_of_20 = await store.feature_view("match").all(limit=20).to_pandas()
```

## Data quality
Alinged will make sure all the different features gets formatted as the correct datatype.
In addition will aligned also make sure that the returend features aligne with defined constraints.

```python
@feature_view(...)
class TitanicPassenger:

    ...

    age = (
        Float()
            .is_required()
            .lower_bound(0)
            .upper_bound(110)
    )
    sibsp = Int32().lower_bound(0, is_inclusive=True)
```

Then since our feature view have a `is_required` and a `lower_bound`, will the `.validate(...)` command filter out the entites that do not follow that behavior.

```python
from aligned.validation.pandera import PanderaValidator

df = await store.model("titanic_model").features_for({
    "passenger_id": [1, 50, 110]
}).validate(
    PanderaValidator()  # Validates all features
).to_pandas()
```

## Feature Server

You can define how to serve your features with the `FeatureServer`. Here can you define where you want to load, and potentially write your features to.

By default will it `aligned` look for a file called `server.py`, and a `FeatureServer` object called `server`. However, this can be defined manually as well.

```python
from aligned import RedisConfig, FileSource
from aligned.schemas.repo_definition import FeatureServer

store = FileSource.json_at("feature-store.json")

server = FeatureServer.from_reference(
    store,
    RedisConfig.localhost()
)
```

Then run `aligned serve`, and a FastAPI server will start. Here can you push new features, which then transforms and stores the features, or just fetch them.

## Stream Worker

You can also setup stream processing with a similar structure. However, here will a `StreamWorker` be used.

by default will `aligned` look for a `worker.py` file with an object called `worker`. An example would be the following.

```python
from aligned import RedisConfig, FileSource
from aligned.schemas.repo_definition import FeatureServer

store = FileSource.json_at("feature-store.json")

server = FeatureServer.from_reference(
    store,
    RedisConfig.localhost()
)
```
