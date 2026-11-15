"""Show the original labeled T-LESS view separately from the new simulation."""

import json
from pathlib import Path

import cv2
import numpy as np
from prepare_scene import read_ply


def build(root):
    root = Path(root)
    provenance = json.loads((root / "source/provenance.json").read_bytes())
    scene = root / "source/test_primesense" / f"{provenance['scene']:06}"
    frame, index = provenance["image_id"], provenance["gt_index"]
    camera = json.loads((scene / "scene_camera.json").read_bytes())[str(frame)]
    gt = json.loads((scene / "scene_gt.json").read_bytes())[str(frame)][index]
    info = json.loads((scene / "scene_gt_info.json").read_bytes())[str(frame)][index]
    assert gt["obj_id"] == 13 and camera["depth_scale"] > 0
    k = np.array(camera["cam_K"]).reshape(3, 3)
    rgb = cv2.imread(str(scene / f"rgb/{frame:06}.png"))
    raw = cv2.imread(str(scene / f"depth/{frame:06}.png"), cv2.IMREAD_UNCHANGED)
    mask = (
        cv2.imread(str(scene / f"mask_visib/{frame:06}_{index:06}.png"), cv2.IMREAD_GRAYSCALE) > 0
    )
    vertices, _ = read_ply(root / "assets/obj_000013.ply")
    r = np.array(gt["cam_R_m2c"]).reshape(3, 3)
    t = np.array(gt["cam_t_m2c"]) * 0.001
    model = vertices @ r.T + t
    uvw = model @ k.T
    uv = uvw[:, :2] / uvw[:, 2, None]
    center = uv.min(0)
    far = uv.max(0)
    measured = np.array(info["bbox_obj"])
    expected = np.r_[center, far - center]
    # BOP pixel boxes enclose the rendered silhouette, so a small raster difference is expected.
    assert np.max(np.abs(expected - measured)) < 5, (expected, measured)
    view = rgb.copy()
    view[mask] = (view[mask] * 0.5 + [30, 210, 80] * np.ones(3) * 0.5).astype(np.uint8)
    x, y, w, h = info["bbox_obj"]
    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 180, 255), 1, cv2.LINE_AA)
    axes = np.vstack([t, *[t + 0.025 * r[:, axis] for axis in range(3)]]) @ k.T
    axes = np.rint(axes[:, :2] / axes[:, 2, None]).astype(int)
    for axis, color in enumerate([(40, 40, 240), (40, 220, 40), (240, 80, 40)]):
        cv2.line(view, tuple(axes[0]), tuple(axes[axis + 1]), color, 2, cv2.LINE_AA)
    cv2.rectangle(view, (0, 0), (view.shape[1], 62), (24, 30, 40), -1)
    cv2.putText(
        view,
        "T-LESS 13 / original pose ground truth",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        "Source view; not the new simulated bin",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (140, 215, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(root / "source-review.png"), view)
    depth = raw.astype(float) * camera["depth_scale"] * 0.001
    yy, xx = np.nonzero((raw > 0) & np.isfinite(depth))
    yy, xx = yy[::8], xx[::8]
    z = depth[yy, xx]
    points = np.c_[(xx - k[0, 2]) * z / k[0, 0], (yy - k[1, 2]) * z / k[1, 1], z]
    colors = rgb[yy, xx, ::-1]
    model = model[::3]
    points = np.vstack([points, model])
    colors = np.vstack([colors, np.tile([200, 90, 235], (len(model), 1))])
    with (root / "source-cloud.ply").open("w", encoding="ascii", newline="\n") as out:
        out.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for p, c in zip(points, colors):
            out.write(" ".join([*(f"{v:.7f}" for v in p), *(str(int(v)) for v in c)]) + "\n")
    metrics = dict(
        schema="tless_ground_truth_view_v1",
        scene=provenance["scene"],
        image_id=frame,
        gt_index=index,
        obj_id=13,
        units="m",
        frame="tless_camera",
        original_bbox_xywh=measured.tolist(),
        projected_model_bbox_xywh=expected.tolist(),
        maximum_bbox_difference_px=float(np.max(np.abs(expected - measured))),
        visible_fraction=info["visib_fract"],
        visible_depth_valid=int(np.count_nonzero(mask & (raw > 0))),
        visible_mask_pixels=int(mask.sum()),
        model_overlay="purple; ground truth, not estimated",
        grasp_ground_truth_available=False,
        simulation_scene_is_original_dataset_scene=False,
    )
    (root / "source-review.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(build(Path(__file__).parent), indent=2))
