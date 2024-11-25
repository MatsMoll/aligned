import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from aligned import ContractStore
from aligned.schemas.derivied_feature import DerivedFeature
from aligned.schemas.feature import Feature, FeatureLocation, FeatureReference
from aligned.schemas.feature_view import CompiledFeatureView
from aligned.schemas.model import Model


@dataclass
class ModelHaveNeededFeaturesCheck:
    model_name: str
    was_unchecked: bool
    missing_features: list[FeatureReference]
    contacts: list[str] | None = None

    @property
    def is_ok(self) -> bool:
        return not self.missing_features

    def as_markdown(self) -> str:
        if self.is_ok:
            return f"Model `{self.model_name}` has all needed features."

        missing_features = '\n- '.join(
            [f"`{f.name}` at `{f.location.identifier}`" for f in self.missing_features]
        )
        markdown = f"Model `{self.model_name}` is missing features: \n- {missing_features}\n\n"
        if self.contacts:
            contacts = '\n- '.join(self.contacts)
            markdown += f"Contacts: {contacts}"

        return markdown


def feature_exist(feature: FeatureReference, store: ContractStore) -> bool:

    loc = feature.location
    if loc.location_type == 'model':
        model = store.model(loc.name).model
        all_features = model.predictions_view.full_schema
    else:
        all_features = store.feature_view(loc.name).view.full_schema

    for f in all_features:
        if f.name == feature.name:
            return True
    return False


async def check_exposed_model_have_needed_features(
    store: ContractStore, model: Model
) -> ModelHaveNeededFeaturesCheck:

    if not model.exposed_model:
        return ModelHaveNeededFeaturesCheck(
            model_name=model.name, was_unchecked=True, missing_features=[], contacts=model.contacts
        )

    try:
        needed_features = await model.exposed_model.needed_features(store.model(model.name))
    except Exception:
        return ModelHaveNeededFeaturesCheck(
            model_name=model.name, was_unchecked=True, missing_features=[], contacts=model.contacts
        )

    missing_features = []
    for feature in needed_features:
        if not feature_exist(feature, store):
            missing_features.append(feature)

    return ModelHaveNeededFeaturesCheck(
        model_name=model.name, was_unchecked=False, missing_features=missing_features, contacts=model.contacts
    )


async def check_exposed_models_have_needed_features(
    store: ContractStore,
) -> list[ModelHaveNeededFeaturesCheck]:

    return await asyncio.gather(
        *[check_exposed_model_have_needed_features(store, model) for model in store.models.values()]
    )


@dataclass
class PotentialModelDistributionShift:
    model_name: str
    reason: str
    contacts: list[str] | None = None

    def as_markdown(self) -> str:
        markdown = f"Model `{self.model_name}` has potential distribution shift:\n{self.reason}\n\n"
        if self.contacts:
            contacts = '\n- '.join(self.contacts)
            markdown += f"Contacts: {contacts}"

        return markdown


async def check_exposed_models_for_potential_distribution_shift(
    old_store: ContractStore, new_store: ContractStore
) -> list[PotentialModelDistributionShift]:

    distribution_shifts: list[PotentialModelDistributionShift] = []
    for model_name, model in new_store.models.items():
        old_model = old_store.models.get(model_name)
        if not old_model:
            continue

        if not old_model.exposed_model or not model.exposed_model:
            continue

        with suppress(NotImplementedError):
            potential_drift = await model.exposed_model.potential_drift_from_model(old_model.exposed_model)
            if potential_drift:
                distribution_shifts.append(
                    PotentialModelDistributionShift(model_name, potential_drift, model.contacts)
                )

    return distribution_shifts


@dataclass
class TransformationDifference:
    location: FeatureLocation
    added: list[DerivedFeature]
    modified: list[tuple[DerivedFeature, Feature]]
    removed: list[DerivedFeature]


def transformation_diff(old_view: CompiledFeatureView, new_view: CompiledFeatureView):

    old_features = {f.name: f for f in old_view.derived_features}
    new_features = {f.name: f for f in new_view.derived_features}

    added = [feature for feature in new_features.values() if feature.name not in old_features]
    removed = [feature for feature in old_features.values() if feature.name not in new_features]
    modified = []

    for feature in old_features.values():
        if feature.name not in new_features:
            continue

        new_feature = new_features[feature.name]
        if feature.transformation != new_feature.transformation:
            modified.append((feature, new_feature))

    return TransformationDifference(
        location=FeatureLocation.feature_view(new_view.name), added=added, modified=modified, removed=removed
    )


@dataclass
class ModelsImpactedByTransformationChanges:
    model_name: str
    features: list[FeatureReference]
    contacts: list[str] | None = field(default=None)

    def as_markdown(self) -> str:
        features = '\n- '.join([f"`{f.name}` at `{f.location.identifier}`" for f in self.features])
        markdown = f"Model `{self.model_name}` is impacted by transformation changes: \n- {features}\n\n"
        if self.contacts:
            contacts = '\n- '.join(self.contacts)
            markdown += f"Contacts: {contacts}"

        return markdown


def impacted_models_from_transformation_diffs(
    new_store: ContractStore, old_store: ContractStore
) -> list[ModelsImpactedByTransformationChanges]:

    diffs: list[TransformationDifference] = []

    for view in new_store.feature_views.values():
        old_view = old_store.feature_views.get(view.name)

        if not old_view:
            continue

        diffs.append(transformation_diff(old_view, view))

    modified_transformation_features = set()

    for diff in diffs:
        for feature, _ in diff.modified:
            modified_transformation_features.add(feature.as_reference(diff.location))

    impacted_models = []
    for model in new_store.models.values():

        all_input_features = model.features.all_features()
        changes = modified_transformation_features.intersection(all_input_features)

        if not changes:
            continue

        impacted_models.append(
            ModelsImpactedByTransformationChanges(
                model_name=model.name, features=list(changes), contacts=model.contacts
            )
        )

    return impacted_models


@dataclass
class ContractStoreUpdateCheckReport:

    needed_model_input: list[ModelHaveNeededFeaturesCheck]
    model_transformation_changes: list[ModelsImpactedByTransformationChanges]
    potential_distribution_shifts: list[PotentialModelDistributionShift]

    def as_markdown(self) -> str:
        markdown = ''

        for check in self.needed_model_input:
            markdown += check.as_markdown() + '\n------------'

        for check in self.potential_distribution_shifts:
            markdown += check.as_markdown() + '\n------------'

        for check in self.model_transformation_changes:
            markdown += check.as_markdown() + '\n------------'

        if markdown:
            markdown = f"## Potential Issues\n\n{markdown}"
        return markdown
