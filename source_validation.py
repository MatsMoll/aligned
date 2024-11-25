from aligned.feature_store import SourceRequest, FeatureLocation


async def validate_sources_in(views: list[SourceRequest]) -> dict[FeatureLocation, bool]:
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

    results: dict[FeatureLocation, bool] = {}

    for view in views:
        try:
            _ = (await view.source.all_data(view.request, limit=1).to_lazy_polars()).collect()
            results[view.location] = True
        except Exception:
            results[view.location] = False

    return results
