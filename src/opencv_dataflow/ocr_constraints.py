"""Constrained CTC decoding over the existing source-region batch recognizer."""

import copy
import hashlib
import json
import math
import re

import numpy as np

from . import ctc_search
from .ml_common import local_path
from .ocr_batch import RecognizeTextBatch, greedy_ctc
from .parameter_validation import NumericRange, ValidationError


def checked_words(words, p):
    if (
        not isinstance(words, list)
        or len(words) > p["maximum_lexicon_words"]
        or any(not isinstance(w, str) or not w.strip() or len(w) > 256 for w in words)
        or len(set(words)) != len(words)
    ):
        raise ValidationError(
            "Use unique nonblank lexicon words of at most 256 characters", "lexicon"
        )
    if sum(map(len, words)) > p["maximum_lexicon_characters"]:
        raise ValidationError("Lexicon exceeds its character budget", "maximum_lexicon_characters")
    return words


def compile_plan(tokens, p, words, digest):
    words = checked_words(words, p)
    blank = p["blank_index"]
    if not 0 <= blank <= len(tokens):
        raise ValidationError("Blank class is outside the model vocabulary", "blank_index")
    mapping = {i if i < blank else i + 1: token for i, token in enumerate(tokens)}
    alphabet = set(p["allowed_characters"])
    allowed = tuple(i for i, token in mapping.items() if not alphabet or set(token) <= alphabet)
    repertoire = set("".join(mapping[i] for i in allowed))
    if any(not set(word) <= repertoire for word in words):
        raise ValidationError(
            "Lexicon uses characters excluded by alphabet/vocabulary",
            "lexicon",
            "allowed_characters",
        )
    trie, terminals = None, set()
    if words:
        trie = [{}]
        for word in words:
            node = 0
            for char in word:
                if char not in trie[node]:
                    trie[node][char] = len(trie)
                    trie.append({})
                node = trie[node][char]
            terminals.add(node)
    return {
        "settings": copy.deepcopy(p),
        "class_tokens": mapping,
        "allowed_ids": allowed,
        "blank": blank,
        "trie": trie,
        "terminals": terminals,
        "lexicon": frozenset(words),
        "lexicon_sha256": digest,
        "trie_nodes": len(trie) if trie else 0,
    }


class RecognizeConstrainedText(RecognizeTextBatch):
    type_id, title = "ml.recognize_constrained_text", "Recognize constrained text regions"
    project_asset_parameters = (*RecognizeTextBatch.project_asset_parameters, "lexicon_path")
    defaults = {
        **copy.deepcopy(RecognizeTextBatch.defaults),
        "decoder": "CTC-prefix-beam-search",
        "allowed_characters": "",
        "lexicon": [],
        "lexicon_path": "",
        "lexicon_sha256": "",
        "beam_width": 16,
        "token_top_k": 0,
        "maximum_alternatives": 3,
        "confidence_mode": "token_normalized",
        "reject_blank_dominant": True,
        "minimum_greedy_ratio": 0.0,
        "maximum_output_tokens": 128,
        "maximum_candidate_prefixes": 8192,
        "maximum_search_expansions": 1_000_000,
        "maximum_alignment_elements": 1_000_000,
        "maximum_lexicon_words": 4096,
        "maximum_lexicon_characters": 65536,
    }
    parameter_choices = {
        **RecognizeTextBatch.parameter_choices,
        "decoder": ("CTC-greedy", "CTC-prefix-beam-search"),
        "confidence_mode": ("token_normalized", "sequence_probability"),
    }
    parameter_ranges = {
        **RecognizeTextBatch.parameter_ranges,
        "beam_width": NumericRange(1, 256, whole=True),
        "token_top_k": NumericRange(0, 10000, whole=True),
        "maximum_alternatives": NumericRange(1, 16, whole=True),
        "minimum_greedy_ratio": NumericRange(0, 1),
        "maximum_output_tokens": NumericRange(1, 512, whole=True),
        "maximum_candidate_prefixes": NumericRange(1, 65536, whole=True),
        "maximum_search_expansions": NumericRange(1, 10_000_000, whole=True),
        "maximum_alignment_elements": NumericRange(1, 8_000_000, whole=True),
        "maximum_lexicon_words": NumericRange(1, 10000, whole=True),
        "maximum_lexicon_characters": NumericRange(1, 262144, whole=True),
    }
    parameter_help = {
        **RecognizeTextBatch.parameter_help,
        "decoder": "Greedy verifies its raw read; prefix beam searches admissible label sequences.",
        "allowed_characters": "Case-sensitive characters; empty permits all vocabulary tokens.",
        "lexicon": "Exact words; no case folding, weighting or spell correction.",
        "lexicon_path": "Alternative project-relative UTF-8 word lines, requiring exact SHA256.",
        "lexicon_sha256": "Exact lexicon file bytes; inline words are sealed with settings.",
        "beam_width": "Prefixes retained per step; approximate search with exact final rescoring.",
        "token_top_k": "Nonblank classes after alphabet filtering; zero keeps all. Blank stays.",
        "maximum_alternatives": "Alternatives with exact unconditioned label-sequence scores.",
        "confidence_mode": "Exact label mass or token-count root; not calibrated correctness.",
        "minimum_confidence": "Selected score limit, retaining original class mass.",
        "reject_blank_dominant": "Reject words with mass at or below the blank sequence.",
        "minimum_greedy_ratio": "Minimum candidate/greedy mass, capped at one; zero disables.",
        "maximum_output_tokens": "Hard decoded-label length limit; reports omitted extensions.",
        "maximum_candidate_prefixes": "Reject too many temporary prefixes before beam pruning.",
        "maximum_search_expansions": "Bound attempted prefix/class transitions per source region.",
        "maximum_alignment_elements": "Combined T x state work for scoring and alignment.",
        "maximum_lexicon_words": "Distinct word limit; the file also has a 1 MB byte guard.",
        "maximum_lexicon_characters": "Combined lexicon characters; bounds the prefix trie.",
    }

    def validate_parameters(self, parameters=None):
        super().validate_parameters(parameters)
        p = self.parameters if parameters is None else parameters
        for key in ("allowed_characters", "lexicon_path", "lexicon_sha256"):
            if not isinstance(p[key], str) or len(p[key]) > 4096:
                raise ValidationError("Use bounded constraint text", key)
        if type(p["reject_blank_dominant"]) is not bool:
            raise ValidationError("Use a checkbox boolean", "reject_blank_dominant")
        checked_words(p["lexicon"], p)
        if p["lexicon_path"]:
            if p["lexicon"]:
                raise ValidationError("Choose inline words or a lexicon file", "lexicon")
            if not re.fullmatch(r"[0-9a-f]{64}", p["lexicon_sha256"]):
                raise ValidationError("Set the lexicon file's lowercase SHA256", "lexicon_sha256")

    def prepare_decoder(self, tokens):
        p = self.parameters
        words = p["lexicon"]
        raw = None
        if p["lexicon_path"]:
            with local_path(self, p["lexicon_path"]).open("rb") as stream:
                raw = stream.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ValidationError("Lexicon file exceeds 1 MB", "lexicon_path")
            if hashlib.sha256(raw).hexdigest() != p["lexicon_sha256"]:
                raise ValidationError("Lexicon file digest differs", "lexicon_sha256")
            words = raw.decode("utf-8-sig").splitlines()
        digest = hashlib.sha256(
            raw
            if raw is not None
            else json.dumps(words, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()
        return compile_plan(tokens, p, words, digest)

    def decoder_report(self):
        return {
            "decoder": self.parameters["decoder"],
            "confidence_semantics": "exact_ctc_label_sequence_mass_"
            + self.parameters["confidence_mode"],
            "character_semantics": "selected_sequence_viterbi_run_peak_probability",
        }

    def decode_row(self, values, tokens, plan):
        p = plan["settings"]
        logp, rounding = ctc_search.log_scores(
            values, len(tokens) + 1, p["score_activation"], p["maximum_steps"]
        )
        greedy_labels = ctc_search.collapsed(logp.argmax(axis=1), plan["blank"])
        greedy_text, greedy_confidence, _ = greedy_ctc(
            values,
            tokens,
            blank=plan["blank"],
            activation=p["score_activation"],
            maximum_steps=p["maximum_steps"],
        )
        best, alternatives, characters, search = ctc_search.decode(logp, plan, greedy_labels)
        budget = {"used": search["alignment_elements"], "maximum": p["maximum_alignment_elements"]}
        greedy_score = ctc_search.forward_score(logp, greedy_labels, plan["blank"], budget)
        search["alignment_elements"] = budget["used"]
        blank_score = float(logp[:, plan["blank"]].sum())
        text, confidence, ratio = "", None, None
        status, reason = "constraint_miss", "No complete admissible sequence remains in the search"
        if best:
            text, confidence = best["text"], best["confidence"]
            ratio = math.exp(min(0, best["log_probability"] - greedy_score))
            if not text.strip():
                status, reason = "blank", "Blank text"
            elif len(best["token_ids"]) > p["maximum_output_tokens"]:
                status, reason = "length_limit", "Decoded label length exceeds its limit"
            elif any(i not in plan["allowed_ids"] for i in best["token_ids"]):
                status, reason = "alphabet_miss", "Raw greedy text violates the allowed alphabet"
            elif plan["lexicon"] and text not in plan["lexicon"]:
                status, reason = "lexicon_miss", "Raw greedy text is outside the lexicon"
            elif p["reject_blank_dominant"] and best["log_probability"] <= blank_score:
                status, reason = "blank_dominant", "Blank sequence has at least as much model mass"
            elif ratio < p["minimum_greedy_ratio"]:
                status, reason = (
                    "greedy_ratio",
                    "Constrained label mass is too low relative to the greedy read",
                )
            elif confidence < p["minimum_confidence"]:
                status, reason = "low_confidence", "Read confidence below limit"
            else:
                status, reason = "read", ""
        return {
            "text": text,
            "confidence": confidence,
            "characters": characters,
            "status": status,
            "passed": status == "read",
            "fail_reasons": []
            if status == "read"
            else [{"code": "ocr." + status, "message": reason}],
            "alternatives": alternatives,
            "decoding": {
                "greedy_text": greedy_text,
                "greedy_path_confidence": greedy_confidence,
                "greedy_log_probability": greedy_score,
                "blank_log_probability": blank_score if np.isfinite(blank_score) else None,
                "selected_log_probability": best["log_probability"] if best else None,
                "selected_token_ids": best["token_ids"] if best else [],
                "relative_to_greedy": ratio,
                "probability_roundoff_max_error": rounding,
                "lexicon_sha256": plan["lexicon_sha256"],
                "lexicon_words": len(plan["lexicon"]),
                "trie_nodes": plan["trie_nodes"],
                "search": search,
            },
        }

    def finish(self, inputs, reads, info, vocabulary_digest):
        result = super().finish(inputs, reads, info, vocabulary_digest)
        result.outputs["report"]["model"] = "constrained_ocr_report_v1"
        result.outputs["report"]["constraint_failures"] = sum(
            row["status"] not in ("read", "blank", "low_confidence", "skipped") for row in reads
        )
        if not result.passed:
            for row in reads:
                for reason in row.get("fail_reasons", []):
                    if reason not in result.fail_reasons:
                        result.fail_reasons.append(reason)
        return result
