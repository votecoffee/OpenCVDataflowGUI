"""OpenCVDataflowGUI block for a saved classifier recipe."""

from ..core import Block, Port, Result
from ..images import opencv_image
from ..payloads import NO_VALUE, frame
from .graphics import draw_assessment
from .model import SURFACE_TEXTURE, load_classifier, read_manifest


class ClassifyImage(Block):
    project_asset_parameters = ("model",)
    type_id, title, category = "inspect.simple_classify_image", "SimpleClassify", "Inspection"
    block_type, subtype = "Inspection", "Classification"
    parallel_safe = True
    process_inputs_readonly = True
    inputs = (Port("image"),)
    outputs = (
        Port("label", "text"),
        Port("score", "number"),
        Port("confusion", "number"),
        Port("probabilities", "data"),
        Port("overlay", required=False),
    )
    defaults = {"model": "", "render_overlay": True}
    parameter_help = {
        "model": "Project-relative manifest staged by the classifier trainer.",
        "render_overlay": "Show the input, ranked scores and confusion in a result chart. "
                          "Disable to skip rendering; scores do not change.",
    }
    plugin_dialog_label = "Train image classifier…"
    plugin_dialog_modeless = True

    def project_assets(self):
        assets = super().project_assets()
        if self.parameters["model"]:
            path, root, data = read_manifest(self.project_dir, self.parameters["model"])
            model = "surface.npz" if data["profile"] == SURFACE_TEXTURE else "forest.yml"
            assets.append((path.parent / model).relative_to(root).as_posix())
            assets.extend(row["path"] for entry in data["classes"] for row in entry["images"])
        return assets

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_key = None
        self._model = None

    def create_plugin_dialog(self, parent, parameters, context):
        from .qt_training import ClassifierTrainer

        return ClassifierTrainer(parent, parameters, context)

    @property
    def preview_image_port(self):
        return "overlay" if self.parameters["render_overlay"] else None

    filmstrip_output = preview_image_port

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        values = self.parameters if parameters is None else parameters
        if type(values["render_overlay"]) is not bool:
            raise ValueError("render_overlay must be boolean.")

    def process(self, inputs):
        manifest = self.parameters["model"]
        if not manifest or self.project_dir is None:
            raise ValueError("Save the project and train the image classifier before running it")
        key = self.project_dir, manifest
        if key != self._model_key:
            self._model = load_classifier(*key)
            self._model_key = key
        prediction = self._model.predict(opencv_image(inputs["image"]))
        label, score = prediction["label"], prediction["score"]
        image = NO_VALUE
        payloads = {}
        if self.parameters["render_overlay"]:
            image = draw_assessment(opencv_image(inputs["image"]), prediction)
            # Even if input and chart sizes happen to match, calibration and
            # source pixel coordinates do not apply to this dashboard.
            payloads["overlay"] = {
                "spatial": frame(list(image.shape[1::-1]), f"{self.id}:class_scores")
            }
        return Result(
            {**prediction, "overlay": image}, score=score / 100,
            message=f"{label} ({score:.2f}%) · confusion {prediction['confusion']:.2f}%",
            payloads=payloads,
            event_data={"filmstrip_priority": -1} if self.parameters["render_overlay"] else {},
        )


def register(registry):
    if registry.types.get(ClassifyImage.type_id) is not ClassifyImage:
        registry.register(ClassifyImage)
