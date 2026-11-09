"""Calculate completion for the selected image and video roadmap capabilities."""

import argparse
import re
from collections import Counter
from pathlib import Path

CHECKLIST = Path(__file__).resolve().parents[1] / "docs/vision-coverage-checklist.md"
STAGES = {
    "1": "Core and baseline 2D",
    "2": "ML and DNN",
    "3": "3D from camera imagery",
    "4": "Selected contrib",
}
STATES = ("Complete", "Partial", "Planned")
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
        if not re.fullmatch(r"S[1-4]-[0-9]{2,}", item) or item[1] != stage:
            raise ValueError(f"Invalid ID or stage placement: {item}")
        if item in seen or status not in STATES:
            raise ValueError(f"Duplicate ID or invalid status: {item} ({status})")
        seen.add(item)
        counts[stage][status] += 1
    if any(not group for group in counts.values()):
        raise ValueError("Each coverage stage needs at least one row")
    return counts


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
    options.add_argument("--check", action="store_true", help="Check the saved summary")
    options.add_argument("--update", action="store_true", help="Update the saved summary")
    args = parser.parse_args()

    with CHECKLIST.open(encoding="utf-8", newline="") as file:
        raw = file.read()
    eol = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace(eol, "\n")
    if "\r" in text or (eol == "\r\n" and "\n" in raw.replace(eol, "")):
        raise SystemExit("Checklist has inconsistent line endings")
    try:
        table = summary_table(stage_counts(text))
        if text.count(START) != 1 or text.count(END) != 1:
            raise ValueError("Checklist needs one summary marker pair")
        first, remainder = text.split(START)
        if END not in remainder:
            raise ValueError("Checklist summary markers are out of order")
        current, last = remainder.split(END)
    except ValueError as error:
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
