"""Private child entry point for the native RIC interpolator."""

import json
import sys

import cv2
import numpy as np


def main():
    p = json.loads(sys.argv[1])
    # OpenCV 5.0 RIC model propagation shares scratch state across iterations.
    cv2.setNumThreads(1)
    solver = cv2.ximgproc.createRICInterpolator()
    solver.setK(int(p["neighbors"]))
    solver.setAlpha(float(p["alpha"]))
    solver.setSuperpixelMode(getattr(cv2.ximgproc, p["superpixel_mode"].upper()))
    solver.setSuperpixelSize(int(p["superpixel_size"]))
    solver.setSuperpixelNNCnt(int(p["superpixel_neighbors"]))
    solver.setSuperpixelRuler(float(p["superpixel_ruler"]))
    solver.setModelIter(int(p["model_iterations"]))
    solver.setRefineModels(p["refine_models"])
    solver.setMaxFlow(float(p["max_flow"]))
    solver.setUseVariationalRefinement(p["variational_refinement"])
    solver.setUseGlobalSmootherFilter(p["post_smoothing"])
    solver.setFGSLambda(float(p["fgs_lambda"]))
    solver.setFGSSigma(float(p["fgs_sigma"]))

    with np.load("input.npz", allow_pickle=False) as inputs:
        source, target = inputs["source"], inputs["target"]
        # Native RIC internally converts the source to Lab and uses BGR-to-gray
        # for its refinement stages. Supply BGR even for monochrome input.
        if source.ndim == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        if target.ndim == 2:
            target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        if inputs["cost"].size:
            solver.setCostMap(inputs["cost"])
        flow = solver.interpolate(source, inputs["source_points"], target, inputs["target_points"])
    np.save("flow.npy", flow, allow_pickle=False)


if __name__ == "__main__":
    main()
