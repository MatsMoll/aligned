# Aladdin

A feature store simplifying feature managment, serving and quality control.
Describe your features, and the feature store grants your wishes so you become the feature king.

## Feature Views

Write features as the should be, as data models.
Then get code completion and typesafety by referencing them in other features.

This makes the features light weight, data source indipendent, and flexible.

```python
class Match(FeatureView):

    metadata = FeatureViewMetadata(
        name="match",
        description="Features about football matches",
        batch_source=...
    )

    # Raw data
    home_team = Entity(dtype=String())
    away_team = Entity(dtype=String())

    date = EventTimestamp(max_join_with=timedelta(days=365))

    half_time_score = String()
    full_time_score = String().description("the scores at full time, in the format 'home-away'. E.g: '2-1'")


    # Transformed features
    is_liverpool = (home_team == "Liverpool").description("If the home team is Liverpool")

    score_as_array = full_time_score.split("-")

    # Custom pandas df method, which get first and second index in `score_as_array`
    home_team_score = score_as_array.transformed(lambda df: df["score_as_array"].str[0].replace({np.nan: 0}).astype(int))
    away_team_score = score_as_array.transformed(...)

    score_differance = home_team_score - away_team_score
    total_score = home_team_score + away_team_score
```

## Data sources

Aladdin makes handling data sources easy, as you do not have to think about how it is done.
Only define where the data is, and we handle the dirty work.

```python
my_db = PostgreSQLConfig(env_var="DATABASE_URL")

class Match(FeatureView):

    metadata = FeatureViewMetadata(
        name="match",
        description="...",
        batch_source=my_db.table(
            "matches",
            mapping_keys={
                "Team 1": "home_team",
                "Team 2": "away_team",
            }
        )
    )

    home_team = Entity(dtype=String())
    away_team = Entity(dtype=String())
```

### Fast development

Making iterativ and fast exploration in ML is important. This is why Aladdin also makes it super easy to combine, and test multiple sources.

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
match_model = ModelService(
    features=[
        Match.select_all(),

        # Select features with code completion
        LocationFeatures.select(lambda view: [
            view.distance_to_match,
            view.duration_to_match
        ]),
    ]
)
```


## Data Enrichers

In manny cases will extra data be needed in order to generate some features.
We therefore need some way of enriching the data.
This can easily be done with Aladdin's `DataEnricher`s.

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
df = await store.all_for("match", limit=2000).to_df()
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
This json file can be generated by running `aladdin apply`.

### Select multiple feature views

```python
df = await store.features_for({
    "home_team": ["Man City", "Leeds"],
    "away_team": ["Liverpool", "Arsenal"],
}, features=[
    "match:home_team_score",
    "match:is_liverpool",

    "other_features:distance_traveled",
]).to_df()
```

### Model Service

Selecting features for a model is super simple.


```python
df = await store.model("test_model").features_for({
    "home_team": ["Man City", "Leeds"],
    "away_team": ["Liverpool", "Arsenal"],
}).to_df()
```

### Feature View

If you want to only select features for a specific feature view, then this is also possible.

```python
prev_30_days = await store.feature_view("match").previous(days=30).to_df()
sample_of_20 = await store.feature_view("match").all(limit=20).to_df()
```

## Data quality
Aladdin will make sure all the different features gets formatted as the correct datatype.
In this way will there be no incorrect format, value type errors.

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

Then run `aladdin serve`, and a FastAPI server will start. Here can you push new features, which then transforms and stores the features, or just fetch them.
