"""Maximum-cardinality, minimum-cost gated assignment shared by image trackers."""

import numpy as np


def associate_costs(values, allowed):
    """Match finite allowed costs in [0,1]; dummy columns represent unmatched tracks."""
    values, allowed = np.asarray(values, float), np.asarray(allowed, bool)
    if values.ndim != 2 or values.shape != allowed.shape or max(values.shape, default=0) > 128:
        raise ValueError("Association matrices need equal 2D shapes with at most 128 per axis")
    if (
        not np.isfinite(values[allowed]).all()
        or np.any(values[allowed] < 0)
        or np.any(values[allowed] > 1)
    ):
        raise ValueError("Allowed association costs must be finite and normalized to [0,1]")
    if not values.size:
        return {}
    rows, detections = values.shape
    # Dummy columns let every track remain unmatched. Their penalty makes
    # cardinality dominate the sum of normalized, in-gate distances.
    penalty = rows + 1
    costs = np.full((rows, detections + rows), float(penalty))
    costs[:, :detections] = np.where(allowed, values, penalty**2)
    columns = costs.shape[1]
    row_potential, col_potential = np.zeros(rows + 1), np.zeros(columns + 1)
    owner = np.zeros(columns + 1, dtype=int)
    previous = np.zeros_like(owner)
    # One-based shortest augmenting paths. Column zero is the active row's root.
    for row in range(1, rows + 1):
        owner[0] = row
        current = 0
        used = np.zeros(columns + 1, dtype=bool)
        best = np.full(columns + 1, np.inf)
        while True:
            used[current] = True
            active = owner[current]
            available = np.flatnonzero(~used)
            reduced = costs[active - 1, available - 1] - row_potential[active]
            reduced -= col_potential[available]
            improved = reduced < best[available]
            changed = available[improved]
            best[changed] = reduced[improved]
            previous[changed] = current
            following = available[np.argmin(best[available])]
            delta = best[following]
            row_potential[owner[used]] += delta
            col_potential[used] -= delta
            best[available] -= delta
            current = following
            if not owner[current]:
                break
        while current:
            prior = previous[current]
            owner[current] = owner[prior]
            current = prior
    return {
        int(row - 1): col - 1
        for col, row in enumerate(owner[1 : detections + 1], 1)
        if row and allowed[row - 1, col - 1]
    }
