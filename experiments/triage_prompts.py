#!/usr/bin/env python3
"""The three triage prompt variants of the pre-registration, plus the "informed" add-on.

The definitions paraphrase eval/ANNOTATION_GUIDE.md, borderline rules included, so the
LLM baseline knows what the gold labels mean. The fixed text comes first and the pair
last, so Ollama can reuse the evaluated prefix between calls.

  python experiments/triage_prompts.py     # print each variant on an invented pair
"""
from __future__ import annotations

from typing import Any

VARIANTS = ("definitions", "examples", "reasoning")
ANSWER_WORDS = ("COMPLEMENT", "SUBSTITUTE", "NEITHER")
NUM_PREDICT = {"definitions": 12, "examples": 12, "reasoning": 120}

DEFINITIONS = """\
You are a retail category manager. Shoppers often buy the two products below together. Decide how they relate, for one question: should this pair appear on a flyer as "buy one, get the other at 20% off"? Judge from the two names only. Their order does not matter.

Choose exactly one class.

COMPLEMENT: different kinds of product that go together. They are used or eaten together, belong to the same recipe, task or occasion, or are parts made to be combined. Test: would a shopper add the second in addition to the first, because of the first?

SUBSTITUTE: the same kind of product. They serve the same function and differ in colour, pattern, flavour, size, pack, brand or a near-equivalent type, so a discount would shift sales from one to the other. Test: would a shopper pick one instead of the other, or is the second just more of the same?

NEITHER: no flyer logic, for one of two reasons. (a) At least one line is not a product for sale: postage, a fee, a deposit, a voucher, a carrier bag handed out at the till, a manual adjustment. (b) Both are products of different kinds with no specific link in use or occasion; they co-occur only because people shop for several things at once.

Borderline rules:
1. Same collection, different function: COMPLEMENT, because they do different jobs at one occasion.
2. Same function, different design: SUBSTITUTE, even if some shoppers buy both.
3. Parts of one whole, made to be combined: COMPLEMENT.
4. Home decor of different kinds: COMPLEMENT only if the names show a shared style, motif or collection, otherwise NEITHER.
5. Food: COMPLEMENT needs a concrete meal, recipe or use. If the only link is that both are everyday groceries, NEITHER.
6. A product that also contains the other: SUBSTITUTE.
7. A bag sold as a designed product (a patterned jumbo bag, a lunch bag) is a product, not an operational line. Apply the other rules to it.
8. Judge a vague category name by what it plausibly covers."""

# Invented for this prompt; run_triage warns if any of these names turns up in a set.
EXAMPLES = (
    ("Snorkel Mask", "Swim Fins", "COMPLEMENT"),
    ("Ski Gloves Black", "Ski Gloves Navy", "SUBSTITUTE"),
    ("Car Wax", "Dog Biscuits", "NEITHER"),
)

ONE_WORD = "Answer with exactly one word: COMPLEMENT, SUBSTITUTE or NEITHER."
REASONING = ('First write one short sentence of reasoning. Then write a final line of the form '
             '"Answer: X", where X is COMPLEMENT, SUBSTITUTE or NEITHER.')


def instructions(variant: str) -> str:
    """The fixed part of a prompt."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown prompt variant {variant!r}")
    parts = [DEFINITIONS]
    if variant == "examples":
        lines = "\n".join(f'- "{a}" + "{b}": {verdict}' for a, b, verdict in EXAMPLES)
        parts.append(f"Examples:\n{lines}")
    parts.append(REASONING if variant == "reasoning" else ONE_WORD)
    return "\n\n".join(parts)


def _words(words: list[str]) -> str:
    return ", ".join(words) if words else "none"


def analysis_block(cl: dict[str, Any]) -> str:
    """The parser's findings as neutral evidence: it never states a verdict or cl_reject."""
    return (
        "A parser reports:\n"
        f"- Product 1 head noun: {cl['head_a'] or 'unknown'}. Attribute words set aside: {_words(cl['stripped_a'])}.\n"
        f"- Product 2 head noun: {cl['head_b'] or 'unknown'}. Attribute words set aside: {_words(cl['stripped_b'])}.\n"
        f"- Same head noun: {'yes' if cl['same_head'] else 'no'}.\n"
        f"- Shared compound: {cl['compound'] or 'none'}.\n"
        f"- Postage, packaging or other operational term: {cl['stop_term'] or 'none'}.\n"
        "This is automatic evidence only. Decide using the definitions above."
    )


def build_prompt(variant: str, name_1: str, name_2: str, analysis: dict[str, Any] | None = None) -> str:
    """One complete prompt; analysis (a "cl" block) adds the informed-condition evidence."""
    parts = [instructions(variant), f"Product 1: {name_1}\nProduct 2: {name_2}"]
    if analysis is not None:
        parts.append(analysis_block(analysis))
    return "\n\n".join(parts)


if __name__ == "__main__":
    sample = {"head_a": "Lamp", "head_b": "Lamp", "stripped_a": ["red"], "stripped_b": ["blue"],
              "same_head": True, "compound": None, "stop_term": None}
    for name in VARIANTS:
        print(f"=== {name} ===\n{build_prompt(name, 'Desk Lamp Red', 'Desk Lamp Blue', sample)}\n")
