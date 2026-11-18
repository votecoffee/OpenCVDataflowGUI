"""Prime native device ownership before short-lived processing workers start."""

import cv2


def prepare_native_runtime(graph_or_blocks):
    """Run on the scheduling/caller thread, never during project loading.

    HOG's SVM upload and KCF can initialize the primary OpenCL device even for
    CPU-facing inputs. Some Windows drivers abort when that owner is a transient
    Qt worker. Preserve the OpenCL preference; only move its lazy initialization.
    """
    graph = graph_or_blocks if hasattr(graph_or_blocks, "blocks") else None
    if graph is not None and getattr(graph, "_native_runtime_ready", False):
        return
    pending = list(graph.blocks.values()) if graph is not None else list(graph_or_blocks)
    graphs, seen, required = [graph] if graph is not None else [], set(), False
    while pending:
        block = pending.pop()
        if id(block) in seen:
            continue
        seen.add(id(block))
        required |= block.type_id == "sequence.track_kcf" or getattr(
            block, "requires_opencl_context", False
        )
        child = getattr(block, "graph", None)
        if child is not None:
            graphs.append(child)
            pending.extend(child.blocks.values())
    if required:
        cv2.ocl.useOpenCL()
        for child in graphs:
            child._native_runtime_ready = True
