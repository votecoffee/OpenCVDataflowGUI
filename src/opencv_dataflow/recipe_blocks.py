"""Visual recipe selection and field/model outputs."""

import copy

from .core import Block, Port, Result
from .payloads import NO_VALUE, ValuePacket
from .recipes import lookup, reference, select_for_value, unique_ref


class RecipeSelect(Block):
    requires_serial_execution = True
    type_id, title = "recipe.select", "Select recipe"
    category, block_type, subtype = "Recipes", "Recipes", "Selection"
    inputs = (Port("recipe", "data", required=False), Port("value", "any", required=False))
    outputs = (
        Port("recipe", "data"),
        Port("settings", "data"),
        Port("models", "data"),
        Port("value", "any"),
        Port("info", "data"),
    )
    defaults = {"recipe_id": "", "activate_global": False, "rebind": False}
    parameter_help = {
        "recipe_id": "Empty uses the image's pinned revision or active recipe.",
        "activate_global": "Activate requests for future captures; pinned reads stay local.",
        "rebind": "Explicitly allow assigning a different recipe to an already versioned value.",
    }
    preserve_image_sets = True
    flow_primary = ("recipe",)

    def process(self, inputs):
        if self._recipes is None:
            raise ValueError("Recipe selection requires a job catalog")
        value = (
            ValuePacket(inputs["value"], self.input_payloads["value"])
            if "value" in inputs
            else None
        )
        selection = inputs.get("recipe", self.parameters["recipe_id"] or None)
        recipe = select_for_value(self._recipes, selection, value, self.parameters["rebind"])
        ref = reference(recipe)
        pinned_before = unique_ref([value]) if value is not None else None
        if self.parameters["activate_global"] and pinned_before is None:
            if not self._allow_recipe_activation:
                raise ValueError(
                    "Global recipe activation is not allowed inside parallel part inspections"
                )
            self._recipes.activate(ref)
        self.last_recipe_ref = ref
        outputs = {
            "recipe": recipe,
            "settings": recipe["settings"],
            "models": recipe["models"],
            "value": inputs.get("value", NO_VALUE),
            "info": ref,
        }
        base = copy.deepcopy(self.input_payloads.get("value", {}))
        base["recipe_ref"] = ref
        return Result(
            outputs,
            payloads={
                name: copy.deepcopy(base) for name in outputs if outputs[name] is not NO_VALUE
            },
            message=f"{ref['id']} · revision {ref['revision']}",
            event_data={"recipe": ref},
        )


class RecipeField(Block):
    requires_serial_execution = True
    type_id, title = "recipe.field", "Recipe value or model"
    category, block_type, subtype = "Recipes", "Recipes", "Outputs"
    inputs = (Port("recipe", "data", required=False), Port("value", "any", required=False))
    outputs = (Port("value", "data"),)
    defaults = {"path": "settings.threshold", "missing": "error"}
    parameter_choices = {"missing": ("error", "inactive")}
    parameter_help = {
        "path": "Dotted path such as settings.threshold or models.part; value pins the revision."
    }
    preserve_image_sets = True

    def process(self, inputs):
        if self._recipes is None:
            raise ValueError("Recipe outputs require a job catalog")
        value = (
            ValuePacket(inputs["value"], self.input_payloads["value"])
            if "value" in inputs
            else None
        )
        recipe = select_for_value(self._recipes, inputs.get("recipe"), value)
        ref = reference(recipe)
        self.last_recipe_ref = ref
        try:
            output = lookup(recipe, self.parameters["path"])
        except ValueError:
            if self.parameters["missing"] == "error":
                raise
            return Result({"value": NO_VALUE}, message="Recipe field absent")
        payload = {"recipe_ref": ref}
        if isinstance(output, dict) and output.get("model") in (
            "shape_pattern_v1",
            "texture_pattern_v1",
        ):
            from .patterns import model_payload

            payload.update(model_payload(output))
        self.last_recipe_ref = ref
        return Result(
            {"value": output},
            payloads={"value": payload},
            message=f"{ref['id']} r{ref['revision']} · {self.parameters['path']}",
        )
