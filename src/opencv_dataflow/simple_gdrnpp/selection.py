"""Lightweight review selection; never changes pose acceptance or imports ML/graphics."""

import math


def candidate_status(candidate):
    if candidate.get("selection", {}).get("status") == "overlap_suppressed":
        return "overlap_suppressed"
    return (candidate.get("fit") or {}).get("status", "unverified")


def describe_review(report, number=0):
    """0 selects the best accepted (else strongest rejected); 1..64 selects a proposal."""
    candidates = report["candidates"]
    if type(number) is not int or not 0 <= number <= 64:
        raise ValueError("Review candidate must be 0 (best available) or a number in 1..64")
    if number > len(candidates):
        raise ValueError(f"Review candidate {number} is unavailable; {len(candidates)} proposals")
    scores = []
    for c in candidates:
        fit = c.get("fit") or {}
        value = fit.get("coverage", 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Review needs finite measured support")
        scores.append(value)
    accepted = [
        i for i, c in enumerate(candidates) if (c.get("fit") or {}).get("status") == "accepted"
    ]
    best = max(accepted, key=lambda i: scores[i]) if accepted else None
    if number:
        selected, reason = number - 1, "requested_candidate"
    elif best is not None:
        selected, reason = best, "best_depth_accepted"
    elif candidates:
        selected, reason = (
            max(range(len(candidates)), key=lambda i: scores[i]),
            "strongest_rejected",
        )
    else:
        selected, reason = None, "no_candidates"
    c = candidates[selected] if selected is not None else None
    selection = dict(
        schema="simple_gdrnpp_review/v1",
        candidate_index=selected,
        candidate_number=selected + 1 if selected is not None else None,
        candidate_id=c["id"] if c is not None else None,
        status=candidate_status(c)
        if c is not None
        else "no_candidates",
        selection_reason=reason,
        is_best_accepted=selected is not None and selected == best,
        best_accepted_number=best + 1 if best is not None else None,
        candidate_count=len(candidates),
    )
    rows = []
    for i, c in enumerate(candidates):
        fit = c.get("fit") or {}
        rmse = fit.get("rmse_m")
        rows.append(
            dict(
                number=i + 1,
                candidate_id=c["id"],
                status=candidate_status(c),
                support_percent=scores[i] * 100,
                contradiction_percent=fit.get("contradiction", 0.0) * 100,
                rmse_mm=rmse * 1000 if rmse is not None else None,
                selected=i == selected,
            )
        )
        if "selection" in c:
            rows[-1].update(
                depth_status=c["depth_fit"]["status"],
                retained_by=c["selection"].get("retained_by"),
                overlap_percent=(c["selection"].get("overlap", 0.0) * 100),
            )
    if selected is not None and "selection" in candidates[selected]:
        chosen = candidates[selected]
        selection.update(
            depth_status=chosen["depth_fit"]["status"],
            retained_by=chosen["selection"].get("retained_by"),
            overlap_percent=chosen["selection"].get("overlap", 0.0) * 100,
        )
    return selection, rows
