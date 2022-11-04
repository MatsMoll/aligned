# Aligned

A feature store alignes the offline and online features in a ML system. Therefore making it harder to achive a training-serving skew. However, that is only the beginning.

## Feature Views

Write features as the should be, as data models.
Then get code completion and typesafety by referencing them in other features.

This makes the features light weight, data source indipendent, and flexible.

```python
class TitanicPassenger(FeatureView):

    metadata = FeatureViewMetadata(
        name="passenger",
        description="Some features from the titanic dataset",
        batch_source=FileSource.csv_at("titanic.csv"),
        stream_source=HttpStreamSource(topic_name="titanic")
    )

    passenger_id = Entity(dtype=Int32())

    # Input values
    age = (
        Float()
            .description("A float as some have decimals")
            .is_required()
            .lower_bound(0)
            .upper_bound(110)
    )

    name = String()
    sex = String().accepted_values(["male", "female"])
    survived = Bool().description("If the passenger survived")
    sibsp = Int32().lower_bound(0, is_inclusive=True).description("Number of siblings on titanic")
    cabin = String()

    # Creates two one hot encoded values
    is_male, is_female = sex.one_hot_encode(['male', 'female'])

    # Standard scale the age.
    # This will fit the scaler using a data slice from the batch source
    # limited to maximum 100 rows. We can also uese a time constraint if wanted
    scaled_age = age.standard_scaled(limit=100)
```

## Data sources

Alinged makes handling data sources easy, as you do not have to think about how it is done.
Only define where the data is, and we handle the dirty work.

```python
my_db = PostgreSQLConfig(env_var="DATABASE_URL")

class TitanicPassenger(FeatureView):

    metadata = FeatureViewMetadata(
        name="passenger",
        description="Some features from the titanic dataset",
        batch_source=my_db.table(
            "passenger",
            mapping_keys={
                "Passenger_Id": "passenger_id"
            }
        ),
        stream_source=HttpStreamSource(topic_name="titanic")
    )

    passenger_id = Entity(dtype=Int32())
```

### Fast development

Making iterativ and fast exploration in ML is important. This is why Aligned also makes it super easy to combine, and test multiple sources.

```python
my_db = PostgreSQLConfig.localhost()

aws_bucket = AwsS3Config(...)

class SomeFeatures(FeatureView):

    metadata = FeatureViewMetadata(
        name="some_features",
        description="...",
        batch_source=my_db.table("local_features")
    )

    # Some features
    ...

class AwsFeatures(FeatureView):

    metadata = FeatureViewMetadata(
        name="aws",
        description="...",
        batch_source=aws_bucket.file_at("path/to/file.parquet")
    )

    # Some features
    ...
```

## Model Service

Usually will you need to combine multiple features for each model.
This is where a `ModelService` comes in.
Here can you define which features should be exposed.

```python
# Uses the variable name, as the model service name.
# Can also define a custom name, if wanted.
titanic_model = ModelService(
    features=[
        TitanicPassenger.select_all(),

        # Select features with code completion
        LocationFeatures.select(lambda view: [
            view.distance_to_shore,
            view.distance_to_closest_boat
        ]),
    ]
)
```


## Data Enrichers

In manny cases will extra data be needed in order to generate some features.
We therefore need some way of enriching the data.
This can easily be done with Alinged's `DataEnricher`s.

```python
my_db = PostgreSQLConfig.localhost()
redis = RedisConfig.localhost()

user_location = my_db.data_enricher( # Fetch all user locations
    sql="SELECT * FROM user_location"
).cache( # Cache them for one day
    ttl=timedelta(days=1),
    cache_key="user_location_cache"
).lock( # Make sure only one processer fetches the data at a time
    lock_name="user_location_lock",
    redis_config=redis
)


async def distance_to_users(df: DataFrame) -> Series:
    user_location_df = await user_location.load()
    ...
    return distances

class SomeFeatures(FeatureView):

    metadata = FeatureViewMetadata(...)

    latitude = Float()
    longitude = Float()

    distance_to_users = Float().transformed(distance_to_users, using_features=[latitude, longitude])
```


## Access Data

You can easily create a feature store that contains all your feature definitions.
This can then be used to genreate data sets, setup an instce to serve features, DAG's etc.

```python
store = FeatureStore.from_dir(".")

# Select all features from a single feature view
df = await store.all_for("passenger", limit=100).to_df()
```

### Centraliced Feature Store Definition
You would often share the features with other coworkers, or split them into different stages, like `staging`, `shadow`, or `production`.
One option is therefore to reference the storage you use, and load the `FeatureStore` from there.

```python
aws_bucket = AwsS3Config(...)
store = await aws_bucket.file_at("production.json").feature_store()

# This switches from the production online store to the offline store
# Aka. the batch sources defined on the feature views
experimental_store = store.offline_store()
```
This json file can be generated by running `alinged apply`.

### Select multiple feature views

```python
df = await store.features_for({
    "passenger_id": [1, 50, 110]
}, features=[
    "passenger:scaled_age",
    "passenger:is_male",
    "passenger:sibsp"

    "other_features:distance_to_closest_boat",
]).to_df()
```

### Model Service

Selecting features for a model is super simple.


```python
df = await store.model("titanic_model").features_for({
    "passenger_id": [1, 50, 110]
}).to_df()
```

### Feature View

If you want to only select features for a specific feature view, then this is also possible.

```python
prev_30_days = await store.feature_view("match").previous(days=30).to_df()
sample_of_20 = await store.feature_view("match").all(limit=20).to_df()
```

## Data quality
Alinged will make sure all the different features gets formatted as the correct datatype.
In addition will aligned also make sure that the returend features aligne with defined constraints.

```python
class TitanicPassenger(FeatureView):

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
).to_df()
```

## Feature Server

This expectes that you either run the command in your feature store repo, or have a file with a `RepoReference` instance.
You can also setup an online source like Redis, for faster storage.

```python
redis = RedisConfig.localhost()

aws_bucket = AwsS3Config(...)

repo_files = RepoReference(
    env_var_name="ENVIRONMENT",
    repo_paths={
        "production": aws_bucket.file_at("feature-store/production.json"),
        "shadow": aws_bucket.file_at("feature-store/shadow.json"),
        "staging": aws_bucket.file_at("feature-store/staging.json")
        # else generate the feature store from the current dir
    }
)

# Use redis as the online source, if not running localy
if repo_files.selected != "local":
    online_source = redis.online_source()
```

Then run `aligned serve`, and a FastAPI server will start. Here can you push new features, which then transforms and stores the features, or just fetch them.
