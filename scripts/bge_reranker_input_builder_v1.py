"""Local, extractive BGE input views; no retrieval, metadata rules or gold input.

Only over-budget pairs change. Short pairs remain byte-identical. A view is a
source-ordered collection of verbatim spans, NOT replacement answer evidence.
"""
from __future__ import annotations

import math
import re
from collections import Counter

VERSION = "bge_input_v1"
SEPARATOR = "\n...\n"
# Language stop words only: no company, metric, formula or benchmark vocabulary.
STOP = frozenset("a an the and or of for to in on at by with from is are was were be been what which how did does do as it its that this than during would could should".split())


def terms(text):
    return {w for w in re.findall(r"[^\W_]+", text.casefold()) if w not in STOP}


class LocalPairTokenizer:
    """Read tokenizer.json directly; preparation never imports torch/transformers."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def pair_count(self, query, text):
        return len(self.tokenizer.encode(query, text).ids)

    def baseline_visible_text(self, query, text, max_length):
        # Match the old HF fast-tokenizer longest_first truncation, including
        # pair special tokens. Offsets refer to the ORIGINAL second sequence.
        self.tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
        try:
            encoded = self.tokenizer.encode(query, text)
            offsets = [p for p, seq in zip(encoded.offsets, encoded.sequence_ids) if seq == 1 and p[1] > p[0]]
            return text[:max((p[1] for p in offsets), default=0)]
        finally:
            self.tokenizer.no_truncation()


def merge_spans(spans):
    merged = []
    for start, end in sorted(spans):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def render(text, spans):
    return SEPARATOR.join(text[a:b] for a, b in merge_spans(spans))


def atomic_spans(text, tokenizer, max_tokens=160):
    """Keep normal rows/sentences intact; split oversized rows at whitespace.

    Decimals/signs are never re-formatted. Pathological single tokens are kept
    intact and can be omitted by the budget check, never sliced into new values.
    """
    pieces, forced_splits = [], 0
    for line in re.finditer(r"[^\n]+(?:\n|$)", text):
        if tokenizer.count(line.group()) <= max_tokens:
            pieces.append([line.start(), line.end()])
            continue
        for sentence in re.finditer(r".+?(?:[.!?](?=\s+[A-Z])\s+|$)", line.group()):
            start, end = line.start() + sentence.start(), line.start() + sentence.end()
            if tokenizer.count(text[start:end]) <= max_tokens:
                pieces.append([start, end])
                continue
            forced_splits += 1
            words = list(re.finditer(r"\S+\s*", text[start:end]))
            piece_start, piece_end = start, start
            for word in words:
                next_end = start + word.end()
                if piece_end > piece_start and tokenizer.count(text[piece_start:next_end]) > max_tokens:
                    pieces.append([piece_start, piece_end])
                    piece_start = piece_end
                piece_end = next_end
            if piece_end > piece_start:
                pieces.append([piece_start, piece_end])
    return pieces, forced_splits


def build_input(question, source_text, tokenizer, max_length=1024):
    if not question.strip() or not source_text.strip():
        raise ValueError("Nonempty question/source text required")
    if max_length < 32 or tokenizer.pair_count(question, "") > max_length - 16:
        raise ValueError("Query leaves insufficient document budget; never truncate the question")
    original_tokens = tokenizer.pair_count(question, source_text)
    if original_tokens <= max_length:
        return {"text": source_text, "changed": False, "original_pair_tokens": original_tokens,
                "input_pair_tokens": original_tokens, "source_spans": [[0, len(source_text)]],
                "retained_source_chars": len(source_text), "omitted_source_chars": 0,
                "forced_long_row_splits": 0, "selected_windows": [], "reason": "already_fits"}

    atoms, forced_splits = atomic_spans(source_text, tokenizer)
    query_terms = terms(question)
    atom_terms = [terms(source_text[a:b]) for a, b in atoms]
    df = Counter(t for ts in atom_terms for t in ts)
    weights = {t: 1 + math.log((1 + len(atoms)) / (1 + df[t])) for t in query_terms}

    # Small original prefix preserves likely title/header/units without
    # inventing table metadata. It is not guaranteed to be a real table header.
    prefix = []
    for span in atoms:
        trial = merge_spans(prefix + [span])
        if tokenizer.count(render(source_text, trial)) > 96:
            break
        prefix = trial
    if not prefix:
        # An oversized opening paragraph must not consume the entire budget.
        for word in re.finditer(r"\S+\s*", source_text):
            if tokenizer.count(source_text[:word.end()]) > 96:
                break
            prefix = [[0, word.end()]]

    windows = []
    for i, span in enumerate(atoms):
        matched = query_terms & atom_terms[i]
        score = sum(weights[t] for t in matched) / math.sqrt(max(1, len(atom_terms[i])))
        windows.append((score, i, matched))
    windows.sort(key=lambda w: (-w[0], w[1]))
    chosen = prefix if tokenizer.pair_count(question, render(source_text, prefix)) <= max_length else []
    decisions = []
    for score, i, matched in windows:
        if any(a <= atoms[i][0] and b >= atoms[i][1] for a, b in chosen):
            continue
        # Prefer the whole local context, then the intact row/sentence itself.
        # Source order is restored after selection; no financial query rules.
        neighborhood = atoms[max(0, i - 1):i + 2]
        for candidate, kind in ((neighborhood, "anchor_and_neighbors"), ([atoms[i]], "anchor_only")):
            trial = merge_spans(chosen + candidate)
            if trial != chosen and tokenizer.pair_count(question, render(source_text, trial)) <= max_length:
                chosen = trial
                decisions.append({"anchor": atoms[i], "method": kind, "score": score,
                                  "matched_query_terms": sorted(matched)})
                break
    if not chosen:
        raise ValueError("No intact source span fits the pair budget")
    result = render(source_text, chosen)
    final_tokens = tokenizer.pair_count(question, result)
    if final_tokens > max_length:
        raise AssertionError("Pair budget violated")
    retained = sum(b - a for a, b in chosen)
    return {"text": result, "changed": True, "original_pair_tokens": original_tokens,
            "input_pair_tokens": final_tokens, "source_spans": chosen,
            "retained_source_chars": retained, "omitted_source_chars": len(source_text) - retained,
            "forced_long_row_splits": forced_splits, "selected_windows": decisions,
            "reason": "query_anchored_verbatim_windows"}
