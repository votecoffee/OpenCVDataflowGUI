"""Summarize retained validation predictions using the assignments' capture IDs."""

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class HeldOutGroup:
    identifier: str
    members: tuple
    correct: int
    common_error: tuple | None

    @property
    def errors(self):
        return len(self.members) - self.correct


def summarize_groups(assigned, predictions, field):
    """Use the known IDs and original assignment indices from a saved health report."""
    members = defaultdict(list)
    errors = defaultdict(Counter)
    for (class_index, image_index), prediction in predictions.items():
        identifier = getattr(assigned[class_index][image_index], field)
        members[identifier].append((class_index, image_index))
        if class_index != prediction.predicted_index:
            errors[identifier][(class_index, prediction.predicted_index)] += 1

    groups = []
    for identifier, keys in members.items():
        mistakes = errors[identifier]
        most_common = None
        if mistakes:
            expected, predicted = min(mistakes, key=lambda key: (-mistakes[key], key))
            most_common = expected, predicted, mistakes[expected, predicted]
        groups.append(
            HeldOutGroup(
                identifier, tuple(sorted(keys)), len(keys) - sum(mistakes.values()), most_common
            )
        )
    # List the largest sources of measured errors first and make ties reproducible.
    return sorted(groups, key=lambda group: (-group.errors, not group.identifier, group.identifier))
