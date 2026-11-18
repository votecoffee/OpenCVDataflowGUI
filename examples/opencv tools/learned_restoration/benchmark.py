"""Direct-native denoising vs known clean pixels, thin defects and tile seams."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PREPARATION = runpy.run_path(str(Path(__file__).with_name("prepare_model.py")))


def crop(image, roi):
    x, y, width, height = roi
    return image[y : y + height, x : x + width]


def metrics(image, reference, truth):
    error = image.astype(np.float64) - reference
    faint = crop(image, truth["faint_crack_roi"]) < 150
    faint_truth = crop(reference, truth["faint_crack_roi"]) < 150
    strong = crop(image, truth["strong_crack_roi"]) < 100
    strong_truth = crop(reference, truth["strong_crack_roi"]) < 100
    return {
        "mse": float(np.square(error).mean()),
        "mae": float(np.abs(error).mean()),
        "maximum_absolute_error": float(np.abs(error).max()),
        "flat_std": float(crop(image, truth["flat_roi"]).std(dtype=np.float64)),
        "texture_rmse": float(np.sqrt(np.square(crop(error, truth["fine_texture_roi"])).mean())),
        "faint_crack_erased_pixels": int((faint_truth & ~faint).sum()),
        "faint_crack_false_pixels": int((~faint_truth & faint).sum()),
        "strong_crack_erased_pixels": int((strong_truth & ~strong).sum()),
        "strong_crack_false_pixels": int((~strong_truth & strong).sum()),
    }


def torch_parity(raw, net):
    import io

    import torch

    if (
        len(raw) != PREPARATION["WEIGHTS_SIZE"]
        or hashlib.sha256(raw).hexdigest() != PREPARATION["WEIGHTS_SHA256"]
    ):
        raise ValueError("Reference weights differ")
    state = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    blob = np.random.default_rng(4).uniform(0, 1, (1, 1, 53, 67)).astype(np.float32)
    with torch.no_grad():
        signal = torch.from_numpy(blob)
        for index in range(17):
            signal = torch.nn.functional.conv2d(
                signal,
                state[f"model.{2 * index}.weight"],
                state[f"model.{2 * index}.bias"],
                padding=1,
            )
            if index < 16:
                signal = torch.relu(signal)
        reference = (torch.from_numpy(blob) - signal).numpy()
    net.setInput(blob, "input")
    return float(np.max(np.abs(net.forward() - reference)))


def benchmark(weights=None):
    truth = json.loads((HERE / "truth.json").read_bytes())
    model = HERE / ".local/native_whole/model.onnx"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == truth["native_onnx_sha256"]
    net = cv2.dnn.readNetFromONNX(str(model), cv2.dnn.ENGINE_CLASSIC)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    reference = cv2.imread(str(HERE / "assets/reference.png"), cv2.IMREAD_GRAYSCALE)
    report = {}
    for case in ("noisy", "low_noise"):
        current = cv2.imread(str(HERE / "assets" / (case + ".png")), cv2.IMREAD_GRAYSCALE)
        unit = current.astype(np.float32) / 255
        net.setInput(unit[None, None], "input")
        whole = net.forward()[0, 0].copy()
        tiled = {}
        for halo in (0, 17):
            image = np.empty_like(unit)
            for y in range(0, 96, 40):
                for x in range(0, 128, 48):
                    right, bottom = min(128, x + 48), min(96, y + 40)
                    left, top, end, stop = (
                        max(0, x - halo),
                        max(0, y - halo),
                        min(128, right + halo),
                        min(96, bottom + halo),
                    )
                    net.setInput(
                        np.ascontiguousarray(unit[top:stop, left:end][None, None]), "input"
                    )
                    tile = net.forward()[0, 0]
                    image[y:bottom, x:right] = tile[y - top : bottom - top, x - left : right - left]
            tiled[halo] = image
        outputs = {
            "input": current,
            "gaussian": cv2.GaussianBlur(current, (3, 3), 1),
            "median": cv2.medianBlur(current, 3),
            **{
                name: np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
                for name, values in (("learned", whole), ("halo17", tiled[17]), ("halo0", tiled[0]))
            },
        }
        report[case] = {
            "metrics": {name: metrics(image, reference, truth) for name, image in outputs.items()},
            "halo17_maximum_unit_error": float(np.max(np.abs(tiled[17] - whole))),
            "halo0_maximum_unit_error": float(np.max(np.abs(tiled[0] - whole))),
            "halo0_maximum_sample_error": int(
                np.abs(outputs["halo0"].astype(int) - outputs["learned"]).max()
            ),
        }
    result = {
        "opencv": cv2.__version__,
        "model_sha256": truth["native_onnx_sha256"],
        "source_weights_sha256": truth["native_weights_sha256"],
        "cases": report,
        "reference": metrics(reference, reference, truth),
        "scope": "known synthetic pixels; fixed DnCNN-25 grayscale CPU model; no physical accuracy",
    }
    if weights is not None:
        result["torch_maximum_unit_error"] = torch_parity(Path(weights).read_bytes(), net)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, help="Also compare against pinned source tensor arithmetic"
    )
    args = parser.parse_args()
    result = benchmark(args.weights)
    destination = HERE / "benchmark.json"
    raw = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if destination.exists() and destination.read_bytes() != raw:
        raise FileExistsError("Different retained benchmark preserved; review before replacing")
    destination.write_bytes(raw)
    print(destination)
