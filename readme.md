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

    home_team = Entity(dtype=String())
    away_team = Entity(dtype=String())

    date = EventTimestamp(max_join_with=timedelta(days=365))

    is_liverpool = (home_team == "Liverpool").description("If the home team is Liverpool")

    full_time_score = (String()
        .description("the scores at full time, in the format 'home-away'. E.g: '2-1'"))

    half_time_score = String()

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

class AwsFeatures(FeatureView):

    metadata = FeatureViewMetadata(
        name="aws",
        description="...",
        batch_source=aws_bucket.file_at("path/to/file.parquet")
    )
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
        LocationFeatures.select(lambda view: [
            view.distance_to_match,
            view.duration_to_match
        ]),
    ]
)
```


## Access Data

You can easily create a feature store that contains all your feature definitions.
This can then be used to genreate data sets, setup an instce to serve features, DAG's etc.

```python
store = FeatureStore.from_dir(".")

# Select all features from a single feature view
df = await store.all_for("match", limit=2000).to_df()
```

### Select multiple feature views

```python
df = await store.features_for({
    "team_1": ["Crystal Palace FC"],
    "team_2": ["Everton FC"]
}, features=[
    "match:half_time_team_1_score",
    "match:is_liverpool",

    "other_features:distance_traveled",
]).to_df()
```

### Model Service

Selecting features for a model is super simple.


```python
df = await store.model("test_model").features_for({
    "team_1": ["Man City", "Leeds"],
    "team_2": ["Liverpool", "Arsenal"],
}).to_df()
```

### Feature View

If you want to only select features for a specific feature view, then this is also possible.

```python
prev_30_days = await store.feature_view("match").previous(days=30).to_df()
samle_of_20 = await store.feature_view("match").all(limit=20).to_df()
```

## Data quality
Aladdin will make sure all the different features gets formatted as the correct datatype.
In this way will there be no incorrect format, value type errors.
