"""OpenCVDataflowGUI block for a saved classifier recipe."""

from ..core import Block, Port, Result
from ..images import opencv_image
from .model import load_classifier


class ClassifyImage(Block):
    type_id, title, category = "inspect.simple_classify_image", "SimpleClassify", "Inspection"
    block_type, subtype = "Inspection", "Classification"
    parallel_safe = True
    inputs = (Port("image"),)
    outputs = (
        Port("label", "text"),
        Port("score", "number"),
        Port("confusion", "number"),
        Port("probabilities", "data"),
    )
    defaults = {"model": ""}
    parameter_help = {"model": "Project-relative manifest staged by the classifier trainer."}
    plugin_dialog_label = "Train image classifier…"
    plugin_dialog_modeless = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_key = None
        self._model = None

    def create_plugin_dialog(self, parent, parameters, context):
        from .qt_training import ClassifierTrainer

        return ClassifierTrainer(parent, parameters, context)

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
        return Result(prediction, score=score / 100, message=f"{label} ({score:.2f}%)")


def register(registry):
    if registry.types.get(ClassifyImage.type_id) is not ClassifyImage:
        registry.register(ClassifyImage)
