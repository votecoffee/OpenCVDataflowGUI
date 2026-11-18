"""Bounded CTC label-prefix search, independent exact rescoring and Viterbi evidence."""

import math

import numpy as np

from .parameter_validation import ValidationError

NEGATIVE = -float("inf")


def log_scores(values, classes, activation, maximum_steps):
    if values.ndim != 2 or values.shape[1] != classes:
        raise ValidationError("CTC classes must equal vocabulary length plus one blank")
    if not 1 <= values.shape[0] <= maximum_steps or not np.isfinite(values).all():
        raise ValidationError("CTC steps or finite scores are invalid", "maximum_steps")
    data = values.astype(np.float64)
    error = 0.0
    if activation == "softmax":
        shifted = data - data.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True)), error
    if activation != "probability" or np.any(data < 0) or np.any(data > 1):
        raise ValidationError("Use softmax logits or normalized probabilities", "score_activation")
    sums = data.sum(axis=1, keepdims=True)
    error = float(np.abs(sums - 1).max())
    if error > 1e-4:
        raise ValidationError("CTC probabilities must sum to one at each step")
    # Normalize full class rows for floating-point roundoff, before any alphabet filtering.
    data /= sums
    result = np.full(data.shape, NEGATIVE)
    np.log(data, out=result, where=data > 0)
    return result, error


def collapsed(path, blank):
    return tuple(
        int(value)
        for i, value in enumerate(path)
        if value != blank and (i == 0 or value != path[i - 1])
    )


def advance(plan, node, token):
    if plan["trie"] is None:
        return 0
    for char in token:
        node = plan["trie"][node].get(char)
        if node is None:
            return None
    return node


def prefix_search(logp, plan):
    p, blank = plan["settings"], plan["blank"]
    # Entries contain disjoint blank/nonblank ending mass, text and lexicon node.
    beams = {(): [0.0, NEGATIVE, "", 0]}
    evidence = dict(
        expansions=0,
        beam_pruned=0,
        class_pruned=0,
        length_limited=0,
        maximum_active_prefixes=1,
        steps_processed=0,
    )

    def add(target, prefix, ending, mass, text, node):
        if mass == NEGATIVE:
            return
        if prefix not in target:
            if len(target) >= p["maximum_candidate_prefixes"]:
                raise ValidationError(
                    "CTC candidate-prefix budget exceeded", "maximum_candidate_prefixes"
                )
            target[prefix] = [NEGATIVE, NEGATIVE, text, node]
        target[prefix][ending] = float(np.logaddexp(target[prefix][ending], mass))

    for samples in logp:
        classes = [i for i in plan["allowed_ids"] if samples[i] != NEGATIVE]
        if p["token_top_k"] and len(classes) > p["token_top_k"]:
            evidence["class_pruned"] += len(classes) - p["token_top_k"]
            classes = sorted(classes, key=lambda i: (-samples[i], i))[: p["token_top_k"]]
        prospective = (len(classes) + 1) * len(beams)
        if evidence["expansions"] + prospective > p["maximum_search_expansions"]:
            raise ValidationError("CTC search work budget exceeded", "maximum_search_expansions")
        evidence["expansions"] += prospective
        target = {}
        for prefix, (pb, pn, text, node) in beams.items():
            total = float(np.logaddexp(pb, pn))
            add(target, prefix, 0, total + samples[blank], text, node)
            for index in classes:
                repeated = bool(prefix and prefix[-1] == index)
                if repeated:
                    add(target, prefix, 1, pn + samples[index], text, node)
                if len(prefix) >= p["maximum_output_tokens"]:
                    evidence["length_limited"] += 1
                    continue
                token = plan["class_tokens"][index]
                next_node = advance(plan, node, token)
                if next_node is None:
                    continue
                add(
                    target,
                    prefix + (index,),
                    1,
                    (pb if repeated else total) + samples[index],
                    text + token,
                    next_node,
                )
        evidence["maximum_active_prefixes"] = max(evidence["maximum_active_prefixes"], len(target))
        ranked = sorted(target, key=lambda key: (-float(np.logaddexp(*target[key][:2])), key))
        evidence["beam_pruned"] += max(0, len(ranked) - p["beam_width"])
        beams = {key: target[key] for key in ranked[: p["beam_width"]]}
        evidence["steps_processed"] += 1
        if not beams:
            break
    complete = [
        key for key, entry in beams.items() if plan["trie"] is None or entry[3] in plan["terminals"]
    ]
    return complete, evidence


def alignment_states(labels, blank, steps, budget):
    size = 2 * len(labels) + 1
    budget["used"] += steps * size
    if budget["used"] > budget["maximum"]:
        raise ValidationError(
            "CTC alignment/rescoring budget exceeded", "maximum_alignment_elements"
        )
    states = np.full(size, blank, np.int32)
    states[1::2] = labels
    skip = np.zeros(size, bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    return states, skip


def forward_score(logp, labels, blank, budget):
    states, skip = alignment_states(labels, blank, len(logp), budget)
    previous = np.full(len(states), NEGATIVE)
    previous[0] = 0
    for row in logp:
        one = np.r_[NEGATIVE, previous[:-1]]
        two = np.full(len(states), NEGATIVE)
        if len(states) > 2:
            two[2:] = previous[:-2]
        two[~skip] = NEGATIVE
        previous = np.logaddexp(np.logaddexp(previous, one), two) + row[states]
    result = float(np.logaddexp(previous[-1], previous[-2])) if labels else float(previous[0])
    return min(result, 0.0)


def viterbi_characters(logp, labels, plan, budget):
    if not labels:
        return []
    states, skip = alignment_states(labels, plan["blank"], len(logp), budget)
    previous = np.full(len(states), NEGATIVE)
    previous[0] = 0
    back = np.empty((len(logp), len(states)), np.uint8)
    for time, row in enumerate(logp):
        two = np.full(len(states), NEGATIVE)
        two[2:] = previous[:-2]
        two[~skip] = NEGATIVE
        options = np.vstack((previous, np.r_[NEGATIVE, previous[:-1]], two))
        moves = options.argmax(axis=0)
        previous = options[moves, np.arange(len(states))] + row[states]
        back[time] = moves
    state = len(states) - 1 if previous[-1] >= previous[-2] else len(states) - 2
    path = np.empty(len(logp), np.int32)
    for time in range(len(logp) - 1, -1, -1):
        path[time] = state
        state -= int(back[time, state])
    characters = []
    for index, label in enumerate(labels):
        steps = np.flatnonzero(path == 2 * index + 1)
        if not len(steps):
            raise ValidationError("Selected CTC sequence has no complete alignment")
        characters.append(
            {
                "token": plan["class_tokens"][label],
                "class_id": int(label),
                "start_step": int(steps[0]),
                "stop_step": int(steps[-1] + 1),
                "confidence": math.exp(float(logp[steps, label].max())),
            }
        )
    return characters


def decode(logp, plan, greedy_labels):
    p = plan["settings"]
    if p["decoder"] == "CTC-greedy":
        sequences, search = [greedy_labels], {"greedy_only": True}
    else:
        sequences, search = prefix_search(logp, plan)
    budget = {"used": 0, "maximum": p["maximum_alignment_elements"]}
    candidates = []
    for labels in sequences:
        text = "".join(plan["class_tokens"][i] for i in labels)
        score = forward_score(logp, labels, plan["blank"], budget)
        if score == NEGATIVE:
            continue
        confidence = math.exp(
            score / (max(1, len(labels)) if p["confidence_mode"] == "token_normalized" else 1)
        )
        candidates.append(
            {
                "text": text,
                "token_ids": list(labels),
                "log_probability": score,
                "probability": math.exp(score),
                "confidence": confidence,
            }
        )
    candidates.sort(key=lambda row: (-row["log_probability"], tuple(row["token_ids"])))
    best = candidates[0] if candidates else None
    characters = viterbi_characters(logp, tuple(best["token_ids"]), plan, budget) if best else []
    search.update(
        candidates_rescored=len(candidates),
        alignment_elements=budget["used"],
        ranking="exact_label_sequence_probability_within_retained_candidates",
        global_optimum_guaranteed=False,
    )
    return best, candidates[: p["maximum_alternatives"]], characters, search
