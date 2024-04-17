# Aligned

A data managment tool for ML applications.

Similar to how DBT is a data managment tool for business analytics, will Aligned manage ML projects.

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
store = await ContractStore.from_dir(".")
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
- [Feature Store](https://matsmoll.github.io/posts/understanding-the-chaotic-landscape-of-mlops#feature-store)
- [Exposing Models](#exposed-models)


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
    input_features=[
        trips.eucledian_distance,
        trips.number_of_passengers,
        traffic.expected_delay
    ],
    output_source=FileSource.delta_at("titanic_model/predictions")
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
from aligned import FileSource, AwsS3Config, AzureBlobConfig

dir_type: Literal["local", "aws", "azure"] = ...

if dir_type == "aws":
    aws_config = AwsS3Config(...)
    root_directory = aws_config.directory("my-awesome-project")

elif dir_type == "azure":
    azure_config = AzureBlobConfig(...)
    root_directory = azure_config.directory("my-awesome-project")
else:
    root_directory = FileSource.directory("my-awesome-project")


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
unix_formatter = DateFormatter.unix_timestamp(time_unit="us", time_zone="UTC")
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

### Exposed models

Aligned mainly focuses on defining the expected input and output of different models. However, this in itself makes it hard to use the models. This is why Aligned makes it possible to define how our ML models are exposed by setting an `exposed_model` attribute.


```python
from aligned.exposed_model.mlflow import mlflow_server

@model_contract(
    name="eta_taxi",
    exposed_model=mlflow_server(
        host="http://localhost:8000",
    ),
    ...
)
class EtaTaxi:
    trip_id = Int32().as_entity()
    predicted_at = EventTimestamp()
    predicted_duration = trips.duration.as_regression_target()
```

This also makes it possible to get predictions with the following command:

```python
await store.model("eta_taxi").predict_over({
    "trip_id": [...]
}).to_polars()
```

Or store them directly in the `output_source` with something like:

```python
await store.model("eta_taxi").predict_over({
    "trip_id": [...]
}).upsert_into_output_source()
```

Some of the existing implementations are:
- MLFlow Server
- Run MLFLow model in memory
- Ollama completion endpoint
- Ollama embedded endpoint
- Send entities to generic endpoint

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

    number_of_passengers = Int32().is_optional()

    dropoff_latitude = Float()
    dropoff_longitude = Float()

    pickup_latitude = Float()
    pickup_longitude = Float()


freshness = await TaxiDepartures.freshness_in_batch_source()

if freshness < datetime.now() - timedelta(days=2):
    raise ValueError("To old data to create an ML model")
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
            .lower_bound(0)
            .upper_bound(110)
    )
    sibsp = Int32().lower_bound(0).is_optional()
```

Then since our feature view have a `is_optional` and a `lower_bound`, will the `.validate(...)` command filter out the entites that do not follow that behavior.

```python
from aligned.validation.pandera import PanderaValidator

df = await store.model("titanic_model").features_for({
    "passenger_id": [1, 50, 110]
}).validate(
    PanderaValidator()  # Validates all features
).to_pandas()
```

## Contract Store

Aligned collects all the feature views and model contracts in a contract store. You can generate this in a few different ways, and each method serves some different use-cases.

For experimentational use-cases will the `await ContractStore.from_dir(".")` probably make the most sense. However, this will scan the full directory which can lead to slow startup times.

Therefore, it is also possible to manually add the different feature views and contracts with the following.

```python
store = ContractStore.empty()
store.add_feature_view(MyView)
store.add_model(MyModel)
```

This makes it possible to define different contracts per project, or team. As a result, you can also combine differnet stores with.

```python
combined_store = recommendation_store.combined_with(forecasting_store)
```

Lastly, we can also load the all features from a serializable format, such as a JSON file.

```python
await FileSource.json_at("contracts.json").as_contract_store()
```
