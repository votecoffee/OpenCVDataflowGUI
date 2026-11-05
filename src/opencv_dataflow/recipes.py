"""Immutable recipe revisions and job-scoped, thread-safe selection."""

import copy
import hashlib
import json
import re
import threading


def identifier(value):
    if type(value) is int:
        value = str(value)
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 128
    ):
        raise ValueError("Recipe ID must be nonempty text (up to 128 characters) or an integer")
    return value


def reference(recipe):
    return {k: recipe[k] for k in ("id", "revision", "digest")}


def validate_reference(ref):
    if (
        not isinstance(ref, dict)
        or set(ref) != {"id", "revision", "digest"}
        or not isinstance(ref.get("id"), str)
        or type(ref.get("revision")) is not int
        or ref["revision"] < 1
        or not isinstance(ref.get("digest"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", ref["digest"])
    ):
        raise ValueError("Invalid recipe revision reference")
    identifier(ref["id"])
    return ref


def digest(recipe):
    body = {k: v for k, v in recipe.items() if k != "digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def check_model(model):
    from .patterns import validate_model
    from .texture_patterns import check_texture_model

    if isinstance(model, dict) and model.get("model") == "trained_image_v1":
        from .image_assets import decode_image

        decode_image(model)
        return model

    if isinstance(model, dict) and model.get("model") == "shape_pattern_v1":
        return validate_model(model, "shape")
    return check_texture_model(model)


def validate_recipe(recipe):
    if (
        not isinstance(recipe, dict)
        or set(recipe) != {"model", "id", "revision", "digest", "settings", "models"}
        or recipe.get("model") != "vision_recipe_v1"
        or not isinstance(recipe.get("settings"), dict)
        or not isinstance(recipe.get("models"), dict)
    ):
        raise ValueError("Expected a vision_recipe_v1 snapshot with settings and models objects")
    validate_reference(reference(recipe))
    if digest(recipe) != recipe["digest"]:
        raise ValueError("Recipe checksum does not match its content")
    if len(json.dumps(recipe, allow_nan=False).encode()) > 16 * 1024 * 1024:
        raise ValueError("Recipe exceeds 16 MiB")
    for name, model in recipe["models"].items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("Model names must be identifiers without dots or spaces")
        check_model(model)
    return recipe


def lookup(recipe, path):
    if not isinstance(path, str) or not path or len(path) > 256:
        raise ValueError("Recipe paths must be nonempty dotted text up to 256 characters")
    value = recipe
    for part in path.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise ValueError(f"Recipe has no field '{path}'")
    return copy.deepcopy(value)


class RecipeCatalog:
    """Never evict revisions implicitly: accepted queued frames may still reference them."""

    def __init__(self):
        self.lock = threading.RLock()
        self._items = {}
        self.default = None
        self.active = None

    def _install(self, recipe):
        ref = reference(recipe)
        for old in self._items.values():
            if (old["id"], old["revision"]) == (ref["id"], ref["revision"]) and old[
                "digest"
            ] != ref["digest"]:
                raise ValueError("Recipe ID/revision already exists with different content")
        proposed = {**self._items, ref["digest"]: copy.deepcopy(recipe)}
        if (
            len(proposed) > 128
            or sum(len(json.dumps(r).encode()) for r in proposed.values()) > 64 * 1024 * 1024
        ):
            raise ValueError(
                "Recipe catalog exceeds 128 revisions or 64 MiB; no revision was changed"
            )
        self._items = proposed
        return copy.deepcopy(recipe)

    def put(self, name, settings, models=None):
        name = identifier(name)
        with self.lock:
            revision = (
                max((r["revision"] for r in self._items.values() if r["id"] == name), default=0) + 1
            )
            recipe = {
                "model": "vision_recipe_v1",
                "id": name,
                "revision": revision,
                "settings": copy.deepcopy(settings),
                "models": copy.deepcopy(models if models is not None else {}),
            }
            recipe["digest"] = digest(recipe)
            validate_recipe(recipe)
            return self._install(recipe)

    def resolve(self, selection=None):
        with self.lock:
            if selection is None or selection == "":
                selection = self.active
            if selection is None:
                raise ValueError("No active recipe; select one in Recipes > Manage recipes")
            if isinstance(selection, dict):
                if selection.get("model") == "vision_recipe_v1":
                    validate_recipe(selection)
                    selection = reference(selection)
                validate_reference(selection)
                recipe = self._items.get(selection["digest"])
                if recipe is None or reference(recipe) != selection:
                    raise ValueError("Recipe revision is not installed in this job")
            else:
                name = identifier(selection)
                candidates = [r for r in self._items.values() if r["id"] == name]
                if not candidates:
                    raise ValueError(f"Unknown recipe '{name}'")
                recipe = max(candidates, key=lambda r: r["revision"])
            return copy.deepcopy(recipe)

    def activate(self, selection, project_default=False):
        with self.lock:
            ref = reference(self.resolve(selection))
            self.active = ref
            if project_default:
                self.default = copy.deepcopy(ref)
            return copy.deepcopy(ref)

    def active_ref(self):
        with self.lock:
            return copy.deepcopy(self.active)

    def definition(self):
        with self.lock:
            return {
                "model": "recipe_catalog_v1",
                "recipes": copy.deepcopy(list(self._items.values())),
                "default": copy.deepcopy(self.default),
            }

    @classmethod
    def from_dict(cls, data):
        if (
            not isinstance(data, dict)
            or data.get("model") != "recipe_catalog_v1"
            or not isinstance(data.get("recipes"), list)
        ):
            raise ValueError("Invalid recipe catalog")
        catalog = cls()
        for recipe in data["recipes"]:
            validate_recipe(recipe)
            catalog._install(recipe)
        if data.get("default") is not None:
            catalog.activate(data["default"], project_default=True)
        return catalog

    def merge(self, other):
        if self is other:
            return
        incoming = other.definition()
        with self.lock:
            previous = self._items
            try:
                for recipe in incoming["recipes"]:
                    self._install(recipe)
            except Exception:
                self._items = previous
                raise
            if self.default is None and incoming["default"] is not None:
                self.default = copy.deepcopy(incoming["default"])
                if self.active is None:
                    self.active = copy.deepcopy(self.default)

    def copy_runtime_from(self, other):
        active = other.active_ref()
        if active is not None:
            self.resolve(active)
        with self.lock:
            self.active = active


def validate_bindings(block, bindings, model_path):
    if (
        not isinstance(bindings, dict)
        or not bindings.keys() <= block.defaults.keys()
        or any(not isinstance(v, str) or not v or len(v) > 256 for v in bindings.values())
        or not isinstance(model_path, str)
        or len(model_path) > 256
    ):
        raise ValueError("Invalid recipe parameter/model bindings")
    if model_path and not getattr(block, "supports_training", False):
        raise ValueError("Only trainable pattern matchers support recipe model bindings")
    if (bindings or model_path) and (
        block.type_id.startswith(("flow.", "recipe."))
        or block.type_id == "input.counter"
        or hasattr(block, "graph")
        or not getattr(block, "recipe_parameters_supported", True)
    ):
        raise ValueError(
            "Recipe overrides are for processing/acquisition tools, not flow scheduling"
        )


def value_refs(value):
    from .payloads import FlowGroup, ValuePacket, metadata

    payload = metadata(value)
    if "recipe_ref" in payload:
        yield validate_reference(payload["recipe_ref"])
    if isinstance(value, ValuePacket):
        value = value.value
    if isinstance(value, FlowGroup):
        for entry in value.items:
            yield from value_refs(entry)


def unique_ref(values):
    refs = [ref for value in values for ref in value_refs(value)]
    if refs and any(ref != refs[0] for ref in refs):
        raise ValueError(
            "Inputs carry different recipe revisions; coordinate the recipe with the image"
        )
    return refs[0] if refs else None


def apply_recipe(block):
    from .payloads import ValuePacket

    validate_bindings(block, block.recipe_bindings, block.recipe_model)
    primary = next(
        (
            name
            for name, value in block.last_inputs.items()
            if hasattr(value, "ndim") and name not in ("mask", "template", "window")
        ),
        None,
    )
    values = (
        [ValuePacket(block.last_inputs[primary], block.input_payloads[primary])]
        if primary
        else [
            ValuePacket(value, block.input_payloads[name])
            for name, value in block.last_inputs.items()
        ]
    )
    # Routing/grouping may intentionally contain different recipes; only bound tools
    # and matchers require a single execution context.
    bound = bool(block.recipe_bindings or block.recipe_model)
    if not getattr(block, "outcome_passthrough", False) and not block.type_id.startswith("recipe."):
        values = [
            ValuePacket(value, block.input_payloads[name])
            for name, value in block.last_inputs.items()
        ]
    try:
        ref = unique_ref(values)
    except ValueError:
        if bound or (
            not getattr(block, "outcome_passthrough", False)
            and not block.type_id.startswith("recipe.")
        ):
            raise
        ref = None
    if block.ticket_source and any(p.kind in ("image", "depth", "pyramid") for p in block.outputs):
        ref = unique_ref(values) or block._recipes.active_ref() if block._recipes else ref
    block.last_recipe_ref = copy.deepcopy(ref)
    if not bound:
        return
    if block._recipes is None:
        raise ValueError("Recipe bindings require a job recipe catalog")
    recipe = block._recipes.resolve(ref)
    block.last_recipe_ref = reference(recipe)
    overrides = {name: lookup(recipe, path) for name, path in block.recipe_bindings.items()}
    for name, value in overrides.items():
        expected = type(block.defaults[name])
        valid = type(value) in (int, float) if expected is float else isinstance(value, expected)
        if expected in (bool, int):
            valid = type(value) is expected
        if not valid:
            raise ValueError(
                f"Recipe value for '{name}' must match parameter type {expected.__name__}"
            )
    parameters = {**copy.deepcopy(block.parameters), **overrides}
    model = None
    if block.recipe_model:
        if "model" in block.last_inputs:
            raise ValueError("Disconnect the model input or clear the recipe model binding")
        model = lookup(recipe, block.recipe_model)
        block.check_model(model)
    block.last_recipe_parameters = copy.deepcopy(overrides)
    block.parameters = parameters
    if block.recipe_model:
        block.trained_model = model


def select_for_value(catalog, selection, value=None, rebind=False):
    pinned = unique_ref([value]) if value is not None else None
    if pinned is not None and not rebind:
        if selection is not None and selection != "":
            if isinstance(selection, dict):
                requested = (
                    reference(selection)
                    if selection.get("model") == "vision_recipe_v1"
                    else selection
                )
                if requested != pinned:
                    raise ValueError("Recipe request differs from the image's pinned revision")
            elif identifier(selection) != pinned["id"]:
                raise ValueError("Recipe request differs from the image's pinned recipe")
        return catalog.resolve(pinned)
    return catalog.resolve(selection)
