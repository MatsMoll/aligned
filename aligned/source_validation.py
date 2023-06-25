from aligned.feature_store import FeatureViewStore


async def validate_sources_in(views: list[FeatureViewStore]) -> dict[str, bool]:
    """Validateds if the sources can fulfill the needs required by the feature views
    Therefore, this means that the views get their "core features".

    ```
    source = FileSource.parquet_at('test_data/titanic.parquet')

    views = feature_store.views_with_batch_source(source)
    validation = await validate_sources_in(views)

    >>> {'titanic_parquet': True}
    ```

    Args:
        views (list[FeatureViewStore]): The feature views to check

    Returns:
        dict[str, bool]: A dict containing the feature view name and if the source full fill the need
    """

    results: dict[str, bool] = {}

    for view in views:
        try:
            view.feature_filter = set(view.request.feature_names)
            _ = await view.all(limit=1).to_polars()
            results[view.name] = True
        except Exception:
            results[view.name] = False

    return results
