"""Plan classifier validation splits from known specimen and recording IDs."""

from collections import Counter
from dataclasses import dataclass

GROUP_FIELDS = (("specimen_id", "Specimen"), ("recording_id", "Recording"))


@dataclass(frozen=True)
class GroupOverlap:
    label: str
    identifier: str
    members: tuple

    @property
    def message(self):
        return (
            f"{self.label} ID {self.identifier!r} crosses training and validation. "
            "Keep its images in one split."
        )


@dataclass(frozen=True)
class TrainingSplit:
    mode: str
    training: tuple
    validation: tuple
    overlaps: tuple = ()
    error: str | None = None


def _named_groups(examples, keys):
    groups = {}
    for key in keys:
        example = examples[key[0]][key[1]]
        for field, label in GROUP_FIELDS:
            value = getattr(example, field)
            if value:
                groups.setdefault((label, value), []).append(key)
    return {name: members for name, members in groups.items() if len(members) > 1}


def _components(keys, groups):
    parents = dict.fromkeys(keys)

    def root(key):
        while parents[key] is not None:
            parent = parents[key]
            if parents[parent] is not None:
                parents[key] = parents[parent]
            key = parent
        return key

    for members in groups.values():
        for member in members[1:]:
            first, second = root(members[0]), root(member)
            if first != second:
                parents[second] = first
    components = {}
    for key in keys:
        components.setdefault(root(key), []).append(key)
    return tuple(components.values())


def _grouped_validation(keys, groups, fingerprints):
    # The ungrouped case never enters this planner. Known IDs link whole components
    # across classes, including chains through both types of ID.
    components = _components(keys, groups)
    totals = Counter(class_index for class_index, _ in keys)
    targets = {i: max(1, total // 5) for i, total in totals.items()}
    choices = []
    for members in components:
        counts = Counter(i for i, _ in members)
        signature = min((fingerprints[key], key) for key in members)
        choices.append((members, counts, signature))

    selected = set()
    counts = Counter()

    def fits(addition):
        return all(counts[i] + extra < totals[i] for i, extra in addition.items())

    def gain(addition):
        return sum(
            (abs(counts[i] - targets[i]) - abs(counts[i] + extra - targets[i])) / totals[i]
            for i, extra in addition.items()
        )

    while any(not counts[i] for i in totals):
        ranked = (
            (-sum(counts[i] == 0 for i in addition), -gain(addition), signature, index)
            for index, (_, addition, signature) in enumerate(choices)
            if fits(addition) and any(not counts[i] for i in addition)
        )
        best = min(ranked, default=None)
        if best is None:
            break
        members, addition, _ = choices.pop(best[-1])
        selected.update(members)
        counts.update(addition)
    if any(not counts[i] for i in totals):
        return selected, (
            "Could not assign whole capture groups to both training and validation for every "
            "class. Add independent groups or mark whole groups for validation."
        )
    # Once each class has a holdout, an addition's benefit only decreases
    # as validation grows. One pass avoids repeated scans of long filmstrips.
    for members, addition, _ in sorted(choices, key=lambda row: (-gain(row[1]), row[2])):
        if fits(addition) and gain(addition) > 0:
            selected.update(members)
            counts.update(addition)
    return selected, None


def plan_split(assignments, fingerprints):
    """Return the image indices in each class in the order used by the trainer.

    Fingerprints contain only readable samples. Keys are (class index, image index).
    """
    examples = tuple(assignments.values())
    eligible = [[] for _ in examples]
    keys = tuple(sorted(fingerprints))
    for class_index, image_index in keys:
        eligible[class_index].append(image_index)
    groups = _named_groups(examples, keys)
    explicit = any(example.validation for rows in examples for example in rows)
    if explicit:
        selected = {key for key in keys if examples[key[0]][key[1]].validation}
        overlaps = tuple(
            GroupOverlap(label, identifier, tuple(members))
            for (label, identifier), members in groups.items()
            if 0 < sum(key in selected for key in members) < len(members)
        )
        mode, error = "explicit", None
    elif groups:
        selected, error = _grouped_validation(keys, groups, fingerprints)
        mode, overlaps = "grouped", ()
    else:
        selected = set()
        for i, rows in enumerate(eligible):
            rows.sort(key=lambda j: fingerprints[i, j])
            selected.update((i, j) for j in rows[:max(1, len(rows) // 5)])
        mode, overlaps, error = "automatic", (), None
    return TrainingSplit(
        mode,
        tuple(tuple(j for j in rows if (i, j) not in selected) for i, rows in enumerate(eligible)),
        tuple(tuple(j for j in rows if (i, j) in selected) for i, rows in enumerate(eligible)),
        overlaps,
        error,
    )
