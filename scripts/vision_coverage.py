"""Calculate completion for the image, video and point-cloud roadmap capabilities."""

import argparse
import re
from collections import Counter
from pathlib import Path

CHECKLIST = Path(__file__).resolve().parents[1] / "docs/vision-coverage-checklist.md"
SCOPE_MAP = CHECKLIST.with_name("opencv-scope-map.md")
STAGES = {
    "1": "Core and baseline 2D",
    "2": "ML and DNN",
    "3": "3D imagery and point clouds",
    "4": "Selected contrib",
}
STATES = ("Complete", "Partial", "Planned")
ITEM_ID = r"S[1-4]-[0-9]{2,}"
SCOPE_REFERENCE = re.compile(rf"({ITEM_ID})(?:–({ITEM_ID}))?")
START = "<!-- coverage-summary:start -->\n"
END = "<!-- coverage-summary:end -->"


def stage_counts(text):
    counts = {stage: Counter() for stage in STAGES}
    seen = set()
    stage = None
    for line in text.splitlines():
        if line.startswith("## Stage "):
            stage = line.split()[2]
        if not re.match(r"^\|\s*S\d", line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5 or not all(cells):
            raise ValueError(f"Invalid coverage row: {line}")
        item, _, _, status, _ = cells
        if not re.fullmatch(ITEM_ID, item) or item[1] != stage:
            raise ValueError(f"Invalid ID or stage placement: {item}")
        if item in seen or status not in STATES:
            raise ValueError(f"Duplicate ID or invalid status: {item} ({status})")
        seen.add(item)
        counts[stage][status] += 1
    if any(not group for group in counts.values()):
        raise ValueError("Each coverage stage needs at least one row")
    return counts, seen


def check_scope_map(text, item_ids):
    mapped = set()
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 3 or not all(cells):
            raise ValueError(f"Invalid scope map row: {line}")
        if cells[2] == "—":
            continue
        for reference in cells[2].split(","):
            match = SCOPE_REFERENCE.fullmatch(reference.strip())
            if match is None:
                raise ValueError(f"Invalid scope map reference: {reference.strip()}")
            first, last = match.groups()
            if last is None:
                mapped.add(first)
                continue
            lo, hi = int(first[3:]), int(last[3:])
            if first[:3] != last[:3] or not 0 <= hi - lo < len(item_ids):
                raise ValueError(f"Invalid scope map range: {reference.strip()}")
            mapped.update(f"{first[:3]}{number:02d}" for number in range(lo, hi + 1))

    unknown = sorted(mapped - item_ids)
    missing = sorted(item_ids - mapped)
    if unknown:
        raise ValueError(f"Unknown scope map IDs: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"Checklist IDs missing from scope map: {', '.join(missing)}")


def summary_table(counts):
    lines = [
        "| Stage | Complete | Partial | Planned | Total | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    overall = Counter()

    def row(label, count):
        total = sum(count.values())
        complete, partial, planned = (count[state] for state in STATES)
        coverage = complete / total
        return f"| {label} | {complete} | {partial} | {planned} | {total} | {coverage:.1%} |"

    for stage, count in counts.items():
        lines.append(row(f"{stage}. {STAGES[stage]}", count))
        overall.update(count)
    lines.append(row("**Overall**", overall))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--check", action="store_true", help="Verify saved counts and scope map")
    options.add_argument("--update", action="store_true", help="Update the saved summary")
    args = parser.parse_args()

    with CHECKLIST.open(encoding="utf-8", newline="") as file:
        raw = file.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace(eol, "\n")
    if "\r" in text or (eol == "\r\n" and "\n" in raw.replace(eol, "")):
        raise SystemExit("Checklist has inconsistent line endings")
    try:
        counts, item_ids = stage_counts(text)
        if args.check or args.update:
            check_scope_map(SCOPE_MAP.read_text(encoding="utf-8"), item_ids)
        table = summary_table(counts)
        if text.count(START) != 1 or text.count(END) != 1:
            raise ValueError("Checklist needs one summary marker pair")
        first, remainder = text.split(START)
        if END not in remainder:
            raise ValueError("Checklist summary markers are out of order")
        current, last = remainder.split(END)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    expected = f"\n{table}\n\n"
    if args.check and current != expected:
        raise SystemExit("Coverage summary is out of date. Run with --update.")
    if args.update and current != expected:
        output = first + START + expected + END + last
        CHECKLIST.write_text(output, encoding="utf-8", newline=eol)
    print(table)


if __name__ == "__main__":
    main()
