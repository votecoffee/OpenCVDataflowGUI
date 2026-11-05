"""Deterministic text checks for OCR and barcode outputs."""

import string

from .core import Block, Port, Result
from .processing import integer

CHARACTERS = {
    "ascii_digits": string.digits,
    "ascii_letters": string.ascii_letters,
    "ascii_alphanumeric": string.ascii_letters + string.digits,
}


class VerifyText(Block):
    type_id, title = "inspect.verify_text", "Verify decoded text"
    category, block_type, subtype = "Inspection", "Inspection", "Code reading"
    inputs = (Port("text", "data"),)
    outputs = (
        Port("checks", "data"),
        Port("matches", "data"),
        Port("count", "number"),
    )
    defaults = {
        "comparison": "none",
        "expected": "",
        "case_sensitive": True,
        "trim": True,
        "minimum_length": 1,
        "maximum_length": 0,
        "character_set": "any",
        "custom_characters": "",
        "aggregate": "all",
    }
    parameter_choices = {
        "comparison": ("none", "exact", "prefix", "suffix", "contains"),
        "character_set": ("any", *CHARACTERS, "custom"),
        "aggregate": ("all", "any"),
    }
    parameter_help = {
        "trim": "Remove leading and trailing whitespace before checking.",
        "maximum_length": "Zero disables the maximum. Length counts Unicode code points.",
        "custom_characters": "Literal allowed characters for the custom character set.",
        "aggregate": "Require every input string or at least one. An empty list always fails.",
    }

    def process(self, inputs):
        data, p = inputs["text"], self.parameters
        if isinstance(data, str):
            data = [data]
        if not isinstance(data, list) or len(data) > 256:
            raise ValueError("Connect a text string or a list of up to 256 reads")
        texts = [entry.get("text") if isinstance(entry, dict) else entry for entry in data]
        if any(not isinstance(text, str) or len(text) > 65_536 for text in texts):
            raise ValueError("Each read requires a text string of at most 65536 characters")
        low = integer(p["minimum_length"], "Minimum length", 0, 65_536)
        high = integer(p["maximum_length"], "Maximum length", 0, 65_536)
        if high and high < low:
            raise ValueError("Maximum length must be zero or at least the minimum length")
        charset = p["character_set"]
        allowed = None
        if charset != "any":
            characters = p["custom_characters"] if charset == "custom" else CHARACTERS[charset]
            if not isinstance(characters, str) or not 1 <= len(characters) <= 4096:
                raise ValueError(
                    "Allowed characters must be a nonempty string up to 4096 characters"
                )
            allowed = set(characters)
        expected = p["expected"]
        if not isinstance(expected, str) or len(expected) > 65_536:
            raise ValueError("Expected value must be a string up to 65536 characters")
        if p["trim"]:
            expected = expected.strip()
        if not p["case_sensitive"]:
            expected = expected.casefold()
        compare = {
            "none": lambda value: True,
            "exact": lambda value: value == expected,
            "prefix": lambda value: value.startswith(expected),
            "suffix": lambda value: value.endswith(expected),
            "contains": lambda value: expected in value,
        }[p["comparison"]]
        checks, matches = [], []
        for index, text in enumerate(texts):
            value = text.strip() if p["trim"] else text
            reasons = []
            if len(value) < low:
                reasons.append("text_too_short")
            if high and len(value) > high:
                reasons.append("text_too_long")
            if allowed is not None and not allowed.issuperset(value):
                reasons.append("text_disallowed_character")
            if not compare(value if p["case_sensitive"] else value.casefold()):
                reasons.append("text_expectation_mismatch")
            checks.append(
                {"index": index, "text": value, "passed": not reasons, "reasons": reasons}
            )
            if not reasons:
                matches.append(value)
        count = len(matches)
        passed = bool(matches) and (p["aggregate"] == "any" or count == len(texts))
        return Result(
            {"checks": checks, "matches": matches, "count": count},
            passed=passed,
            score=count / len(texts) if texts else 0.0,
            message=f"{count} of {len(texts)} reads meet text rules",
            fail_reasons=[]
            if passed
            else ["text_verification_failed" if texts else "no_text_reads"],
        )
