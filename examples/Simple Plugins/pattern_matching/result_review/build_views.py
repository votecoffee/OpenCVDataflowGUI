"""Export selected fits from saved public recipes, without retuning them."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from opencv_geometric_pattern_match import load_recipe
from opencv_geometric_pattern_match.image_input import prepare_image, working_to_source
from opencv_geometric_pattern_match.locator import review_search
from opencv_geometric_pattern_match.result_review import save_candidate_review

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.processing import gray_image


def build(target):
    root = Path(__file__).resolve().parents[1]
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    receipt = {}
    for kind in ("asymmetric_brackets", "visa_pcb2"):
        folder = root / kind
        graph = Graph.load(folder / "project.json", builtin_registry())
        try:
            block = next(b for b in graph.blocks.values() if b.type_id == "pattern.ogpm_recipe")
            recipe = load_recipe(folder / block.parameters["recipe_path"])
            for index in range(2):
                graph.run()
                if block.state not in ("pass", "fail"):
                    raise ValueError(block.error)
                native = gray_image(block.last_inputs["image"], (np.uint8, np.uint16))
                image = prepare_image(native, recipe.pattern.input_scale)
                review = review_search(
                    image, recipe.pattern, recipe.search, roi=recipe.search_roi, capture_fit=True
                )
                mapping = working_to_source(native.shape, image.shape)
                # Lowest-score accepted fit illustrates remaining mismatches;
                # keep the saved recipe and its acceptance settings unchanged.
                candidates = [("accepted", review.matches[-1])] if review.matches else []
                rejected = next((r for r in review.rejected if r.reason == "score_cutoff"), None)
                if rejected is not None:
                    candidates.append(("rejected", rejected))
                for status, candidate in candidates:
                    name = f"{kind}_{index}_{status}"
                    save_candidate_review(
                        target / name,
                        image,
                        recipe.pattern,
                        candidate,
                        recipe.search,
                        working_to_source=mapping,
                    )
                    receipt[name] = {
                        "project": (folder / "project.json").relative_to(root).as_posix(),
                        "frame_index": index,
                        "accepted_count": len(review.matches),
                        "rejection_counts": review.rejection_counts,
                        "recipe_sha256": hashlib.sha256(
                            (folder / block.parameters["recipe_path"]).read_bytes()
                        ).hexdigest(),
                        "native_pixels_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
                        "working_pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "native_size": list(native.shape[1::-1]),
                        "working_size": list(image.shape[1::-1]),
                    }
                print(kind, index, len(review.matches), "accepted")
        finally:
            graph.close()
    (target / "receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="New review directory; existing directories are refused"
    )
    build(parser.parse_args().output)
