#!/usr/bin/env python3
"""Task B: flyer copy under three conditions, for 30 gold-complement bundles.

  raw      one call for all three languages, used as it comes
  guarded  reuses the raw output; each failing language gets targeted repair calls
           (max 2 retries), then pipeline.template_copy for that language
  delex    the LLM writes with [A] and [B] and never sees the product names; same repair;
           code fills in the exact names; final checks on the filled text; template last

Every call, every try and every final text goes to experiments/results/copy_runs.jsonl.
A rerun replays the calls already logged and only makes the missing ones. Readers take the
last record per key, since --retry-errors can change what was derived from a failed call.

  python experiments/run_copy.py
  python experiments/run_copy.py --dev-smoke --max-calls 8
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cl_features  # noqa: E402
import common  # noqa: E402
import copy_checks as checks  # noqa: E402

LANGS = ("de", "en", "el")
CONDITIONS = ("raw", "guarded", "delex")
ALL_CONDITIONS = (*CONDITIONS, "delex_context")
LANGUAGE_NAMES = {"de": "German", "en": "English", "el": "Modern Greek"}
LANGUAGE_RULES = {"de": "Write in German and address the reader with du.", "en": "Write in English.",
                  "el": "Write in Modern Greek, in Greek script."}
DISCOUNT = 20
MAX_RETRIES = 2
TEMPERATURE = 0.4     # as in the pipeline; a retry's new seed would change nothing at 0
FIRST_PREDICT = 400   # as in the pipeline, for all three languages
REPAIR_PREDICT = 200  # one language only
BUNDLE_SEED = 20261106
MAX_CONSECUTIVE_ERRORS = 3


def runs_path() -> Path:
    return common.results_dir() / "copy_runs.jsonl"


def bundles_path() -> Path:
    return common.results_dir() / "copy_bundles.json"


# --- Bundles ---------------------------------------------------------------------

def anchor_and_pull(pair: dict[str, Any]) -> tuple[str, str]:
    """Anchor = the higher price; without prices (or at equal price), product a."""
    price_a, price_b = pair.get("price_a"), pair.get("price_b")
    if price_a is not None and price_b is not None and price_b > price_a:
        return pair["b"], pair["a"]
    return pair["a"], pair["b"]


def _bundle(number: int, pair: dict[str, Any]) -> dict[str, Any]:
    anchor, pull = anchor_and_pull(pair)
    return {"id": f"b{number:02d}", "set": pair["set"], "pair_id": pair["pair_id"], "anchor": anchor, "pull": pull,
            "pct": DISCOUNT, "lift": pair.get("lift")}


def select_bundles(n_per_set: int = 10, seed: int = BUNDLE_SEED) -> list[dict[str, Any]]:
    """10 highest-lift gold complements from t1 and t2, 10 seeded from t3 (which has no lift)."""
    chosen: list[dict[str, Any]] = []
    for set_id in common.TEST_SETS:
        gold = common.read_gold(set_id)
        pairs = [p for p in common.read_pairs(set_id) if gold[p["pair_id"]] == "complement"]
        if len(pairs) < n_per_set:
            print(f"WARNING: {set_id} has only {len(pairs)} gold complements", file=sys.stderr, flush=True)
        if set_id == "t3":
            pairs = sorted(pairs, key=lambda p: p["pair_id"])
            chosen += random.Random(f"{seed}:{set_id}").sample(pairs, min(n_per_set, len(pairs)))
        else:
            chosen += sorted(pairs, key=lambda p: (-(p["lift"] or 0), p["pair_id"]))[:n_per_set]
    return [_bundle(i, pair) for i, pair in enumerate(chosen, start=1)]


def dev_smoke_bundle(pipeline: ModuleType) -> list[dict[str, Any]]:
    """The highest-lift dev pair that passes the rule stage: a likely complement, for testing only."""
    pairs = common.read_pairs("dev")
    analysis = cl_features.analyse_pairs(pairs, common.read_catalog("dev"), pipeline)
    survivors = [p for p in pairs if not analysis[p["pair_id"]]["cl_reject"]]
    best = sorted(survivors, key=lambda p: (-(p["lift"] or 0), p["pair_id"]))[0]
    return [_bundle(1, best)]


# --- Parsing ---------------------------------------------------------------------

def _entry(value: Any) -> tuple[str, str] | None:
    """(title, copy) out of one language's JSON object, or None if either is unusable."""
    if not isinstance(value, dict):
        return None
    lowered = {str(k).lower(): v for k, v in value.items()}
    title, copy = lowered.get("title"), lowered.get("copy")
    if not (isinstance(title, str) and isinstance(copy, str)) or not title.strip() or not copy.strip():
        return None
    return title.strip(), copy.strip()


def _loads(text: str) -> Any:
    start, end = text.find("{"), text.rfind("}")
    if not 0 <= start < end:
        return None
    try:
        return json.loads(text[start:end + 1], strict=False)
    except ValueError:
        return None


def parse_all_languages(text: str) -> dict[str, tuple[str, str]]:
    """The languages readable from a three-language answer.

    The JSON is cut out of plain text. If the whole object does not parse (for example the
    answer was cut off), each language's object is still tried on its own.
    """
    whole = _loads(text)
    lowered = {str(k).lower(): v for k, v in whole.items()} if isinstance(whole, dict) else {}
    out = {}
    for lang in LANGS:
        entry = _entry(lowered.get(lang))
        if entry is None:
            match = re.search(rf'"{lang}"\s*:\s*(\{{.*?\}})', text, re.DOTALL | re.IGNORECASE)
            entry = _entry(_loads(match.group(1))) if match else None
        if entry is not None:
            out[lang] = entry
    return out


def parse_one_language(text: str, lang: str) -> tuple[str, str] | None:
    """The answer to a repair call: {"title","copy"}, also accepted when nested under the language."""
    whole = _loads(text)
    if isinstance(whole, dict):
        lowered = {str(k).lower(): v for k, v in whole.items()}
        return _entry(lowered) or _entry(lowered.get(lang))
    return None


# --- Prompts ---------------------------------------------------------------------

def delex_prompt(pct: int) -> str:
    """The pipeline's prompt, delexicalised: the model sees only the placeholders."""
    return (
        "You write supermarket flyer copy. Return only JSON of the form "
        '{"de":{"title":"","copy":""},"en":{"title":"","copy":""},"el":{"title":"","copy":""}}. '
        "de = German (address the reader with du), en = English, el = Modern Greek in Greek script. "
        "The two products are hidden behind placeholders: [A] is the full-price item and [B] is the discounted item. "
        "title: at most 6 words, with no placeholder. copy: one catchy sentence, at most 25 words, that says "
        f"[B] is {pct}% off when bought with [A]. Write [A] and [B] literally, exactly once each, in every copy, "
        "with no article directly before them. Do not invent prices, product details or other offers."
    )


def delex_context_prompt(anchor: str, pull: str, pct: int) -> str:
    """Delexicalised output with semantic context: the model sees the product names but emits [A] and [B]."""
    shown = json.dumps({"A_full_price": anchor, "B_discounted": pull, "discount_pct": pct}, ensure_ascii=False)
    return (
        "You write supermarket flyer copy. Return only JSON of the form "
        '{"de":{"title":"","copy":""},"en":{"title":"","copy":""},"el":{"title":"","copy":""}}. '
        "de = German (address the reader with du), en = English, el = Modern Greek in Greek script. "
        "Use the bundle below for thematic context only; in the copy itself, represent the full-price item as [A] "
        "and the discounted item as [B] so code can insert the exact catalog strings afterwards. "
        "title: at most 6 words, with no placeholder and no product name. copy: one catchy sentence, at most 25 words, "
        f"that says [B] is {pct}% off when bought with [A]. Write [A] and [B] literally, exactly once each, in every "
        f"copy, with no article directly before them. Do not invent prices or other offers.\nBundle: {shown}"
    )


def _problem_lines(results: dict[str, dict[str, Any]], lang: str, names: tuple[str, ...], pct: int,
                   delex: bool) -> list[str]:
    """One plain instruction per failed check, naming exactly what went wrong."""
    lines = []
    for check in checks.failed_ids(results):
        detail = results[check]["detail"]
        if check == "parse":
            lines.append("Your last answer could not be read. Return only the JSON object.")
        elif check == "names" and delex:
            lines.append(f"{detail}. The copy must contain [A] exactly once and [B] exactly once, with the "
                         "brackets. The title must contain neither.")
        elif check == "names":
            lines.append('The copy must contain both product names exactly as written, not translated or '
                         f'shortened: "{names[0]}" and "{names[1]}".')
        elif check == "number":
            lines.append(f"{detail}. State the discount as {pct}% in the copy and use no other number.")
        elif check == "sentence":
            lines.append(f"The copy has {detail}. It must be exactly one sentence.")
        elif check == "length":
            lines.append(f"{detail}. Keep the title to at most 6 words and the copy to at most 25 words.")
        elif check == "language":
            lines.append(f"{detail}. {LANGUAGE_RULES[lang]}")
        elif check == "address":
            lines.append(f'{detail}. Address the reader with "du" or an imperative such as "Kaufe", never with '
                         '"Sie", "Ihr" or "Ihnen".')
        elif check == "claim":
            lines.append(f"{detail}. Remove it: the discount is the only offer.")
        elif check == "oov":
            lines.append(f"{detail}. Replace those words with common Greek words.")
    return lines


def repair_prompt(bundle: dict[str, Any], lang: str, entry: tuple[str, str] | None,
                  results: dict[str, dict[str, Any]], names: tuple[str, ...], delex: bool,
                  delex_context: bool = False) -> str:
    pct = bundle["pct"]
    if delex and delex_context:
        shown = json.dumps({"A_full_price": names[0], "B_discounted": names[1], "discount_pct": pct},
                           ensure_ascii=False)
        context = ("Use the bundle for thematic context only; in the copy itself, represent the full-price item as "
                   f"[A] and the discounted item as [B]. The copy says [B] is {pct}% off when bought with [A]. "
                   "Write [A] and [B] literally, exactly once each, in the copy, with no article directly before "
                   f"them, and neither in the title.\nBundle: {shown}")
    elif delex:
        context = ("The two products are hidden behind placeholders: [A] is the full-price item and [B] is the "
                   f"discounted item. The copy says [B] is {pct}% off when bought with [A]. Write [A] and [B] "
                   "literally, exactly once each, in the copy, with no article directly before them, and neither "
                   "in the title.")
    else:
        shown = json.dumps({"anchor_full_price": names[0], "discounted_item": names[1], "discount_pct": pct},
                           ensure_ascii=False)
        context = (f"The copy says the discounted item is {pct}% off when bought with the full-price item. "
                   f"Use the product names as given.\nBundle: {shown}")
    previous = f'Previous title: {entry[0]}\nPrevious copy: {entry[1]}\n' if entry else ""
    problems = "\n".join(f"- {line}" for line in _problem_lines(results, lang, names, pct, delex))
    return (
        f"You write supermarket flyer copy in {LANGUAGE_NAMES[lang]}. {LANGUAGE_RULES[lang]} "
        'Fix the text below and return only JSON of the form {"title":"","copy":""}. '
        "title: at most 6 words. copy: one catchy sentence, at most 25 words. "
        f"Do not invent prices or other offers. {context}\n{previous}Problems to fix:\n{problems}"
    )


# --- The run log -----------------------------------------------------------------

class RunLog:
    """Append-only log. Calls already in it are replayed instead of being made again."""

    def __init__(self, path: Path, retry_errors: bool = False, max_calls: int | None = None) -> None:
        self.path = path
        self.calls: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.notes: dict[tuple[Any, ...], dict[str, Any]] = {}
        for record in common.read_jsonl(path):
            if record["kind"] == "call":
                self.calls[self._call_key(record)] = record
            else:
                self.notes[self._note_key(record)] = record
        if retry_errors:
            self.calls = {k: v for k, v in self.calls.items() if not v.get("error")}
        self.max_calls = max_calls
        self.calls_made = 0
        self.consecutive_errors = 0

    @staticmethod
    def _call_key(r: dict[str, Any]) -> tuple[Any, ...]:
        return r["bundle_id"], r["condition"], r["lang"], r["try"]

    @staticmethod
    def _note_key(r: dict[str, Any]) -> tuple[Any, ...]:
        return r["kind"], r["bundle_id"], r["condition"], r["lang"], r.get("try")

    def call(self, bundle_id: str, condition: str, lang: str, try_no: int, prompt: str, seed: int,
             num_predict: int) -> dict[str, Any] | None:
        """The logged or new response. None only when a --max-calls budget is used up."""
        key = (bundle_id, condition, lang, try_no)
        if key in self.calls:
            if self.calls[key]["prompt"] != prompt:
                sys.exit(f"{self.path}: {key} was run with a different prompt; move the file away to restart.")
            return self.calls[key]
        if self.max_calls is not None and self.calls_made >= self.max_calls:
            return None
        reply = common.generate(prompt, num_predict=num_predict, temperature=TEMPERATURE, seed=seed)
        self.calls_made += 1
        record = {"kind": "call", "bundle_id": bundle_id, "condition": condition, "lang": lang, "try": try_no,
                  "seed": seed, "prompt": prompt, "raw": reply["response"],
                  "prompt_eval_count": reply["prompt_eval_count"], "eval_count": reply["eval_count"],
                  "total_duration": reply["total_duration"], "load_duration": reply["load_duration"],
                  "wall_ms": reply["wall_ms"], "attempts": reply["attempts"]}
        if reply["error"]:
            record["error"] = reply["error"]
        common.append_jsonl(self.path, record)
        self.calls[key] = record
        self.consecutive_errors = self.consecutive_errors + 1 if reply["error"] else 0
        if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            sys.exit(f"{self.consecutive_errors} failed calls in a row: Ollama looks down. "
                     "Fix it, then rerun with --retry-errors.")
        return record

    def note(self, kind: str, **fields: Any) -> None:
        """Log a try or a final text, unless the log already has exactly this content."""
        record = {"kind": kind, **fields}
        key = self._note_key(record)
        if self.notes.get(key) != record:
            common.append_jsonl(self.path, record)
            self.notes[key] = record


def retry_seed(condition: str, lang: str, try_no: int) -> int:
    """A different seed for every retry; the condition and language keep them apart too."""
    return common.SEED + 100 * (ALL_CONDITIONS.index(condition) + 1) + 10 * (LANGS.index(lang) + 1) + try_no


# --- Conditions ------------------------------------------------------------------

def _fill(text: str, names: tuple[str, ...]) -> str:
    return text.replace("[A]", names[0]).replace("[B]", names[1])


def assess(lang: str, entry: tuple[str, str] | None, names: tuple[str, ...], pct: int,
           delex: bool) -> tuple[dict[str, dict[str, Any]], str, str]:
    """Check one text. Returns (results, title, copy); delex text is filled once it passes."""
    title, copy = entry or ("", "")
    if not delex:
        return checks.run_checks(lang, title, copy, names, pct), title, copy
    results = checks.run_checks(lang, title, copy, names, pct, delex=True)
    if checks.failed_ids(results):
        return results, title, copy
    filled_title, filled_copy = _fill(title, names), _fill(copy, names)
    return checks.run_checks(lang, filled_title, filled_copy, names, pct), filled_title, filled_copy


def _final(log: RunLog, bundle: dict[str, Any], condition: str, lang: str, title: str, copy: str, source: str,
           tries: int, results: dict[str, dict[str, Any]], first_failed: list[str]) -> None:
    oov = results.get("oov", {})
    log.note("final", bundle_id=bundle["id"], set=bundle["set"], condition=condition, lang=lang, title=title,
             copy=copy, source=source, tries=tries, failed_checks=checks.failed_ids(results),
             first_try_failed_checks=first_failed, oov_words=checks.oov_words(results),
             strict_oov_words=oov.get("strict_oov_words", []), check_details=results)


def _try_note(log: RunLog, bundle: dict[str, Any], condition: str, lang: str, try_no: int,
              entry: tuple[str, str] | None, results: dict[str, dict[str, Any]]) -> None:
    log.note("try", bundle_id=bundle["id"], condition=condition, lang=lang, **{"try": try_no},  # "try" is a keyword
             title=entry[0] if entry else "", copy=entry[1] if entry else "",
             failed_checks=checks.failed_ids(results), check_details=results)


def finish_raw(log: RunLog, bundle: dict[str, Any], lang: str, entry: tuple[str, str] | None,
               names: tuple[str, ...]) -> None:
    results, title, copy = assess(lang, entry, names, bundle["pct"], delex=False)
    _try_note(log, bundle, "raw", lang, 0, entry, results)
    failed = checks.failed_ids(results)
    _final(log, bundle, "raw", lang, title, copy, "llm" if entry else "none", 1, results, failed)


def repair_loop(log: RunLog, bundle: dict[str, Any], condition: str, lang: str, first: tuple[str, str] | None,
                names: tuple[str, ...], pipeline: ModuleType) -> None:
    """guarded, delex and delex_context: repair a failing language up to twice, then use the template."""
    delex = condition in ("delex", "delex_context")
    delex_context = condition == "delex_context"
    pct = bundle["pct"]
    entry = first
    results, title, copy = assess(lang, entry, names, pct, delex)
    first_failed = checks.failed_ids(results)
    _try_note(log, bundle, condition, lang, 0, entry, results)
    tries = 1
    while checks.failed_ids(results) and tries <= MAX_RETRIES:
        prompt = repair_prompt(bundle, lang, entry, results, names, delex, delex_context=delex_context)
        record = log.call(bundle["id"], condition, lang, tries, prompt, retry_seed(condition, lang, tries),
                          REPAIR_PREDICT)
        if record is None:  # the smoke test's call budget ran out
            break
        entry = parse_one_language(record["raw"], lang)
        results, title, copy = assess(lang, entry, names, pct, delex)
        _try_note(log, bundle, condition, lang, tries, entry, results)
        tries += 1
    if checks.failed_ids(results):
        text = pipeline.template_copy(bundle["anchor"], bundle["pull"], pct, f"{bundle['anchor']}|{bundle['pull']}")[lang]
        title, copy, source = text["title"], text["Copy"], "template"
        results = checks.run_checks(lang, title, copy, names, pct)
    else:
        source = "llm"
    _final(log, bundle, condition, lang, title, copy, source, tries, results, first_failed)


def process_bundle(bundle: dict[str, Any], log: RunLog, pipeline: ModuleType) -> None:
    """All three conditions of one bundle. Both first calls come before any repair call."""
    names = (pipeline.display_name(bundle["anchor"]), pipeline.display_name(bundle["pull"]))
    raw_prompt = pipeline.GemmaCopywriter._prompt(None, bundle["anchor"], bundle["pull"], bundle["pct"])
    raw = log.call(bundle["id"], "raw", "all", 0, raw_prompt, common.SEED, FIRST_PREDICT)
    delex = log.call(bundle["id"], "delex", "all", 0, delex_prompt(bundle["pct"]), common.SEED, FIRST_PREDICT)
    raw_entries = parse_all_languages(raw["raw"]) if raw else {}
    delex_entries = parse_all_languages(delex["raw"]) if delex else {}
    for lang in LANGS:
        finish_raw(log, bundle, lang, raw_entries.get(lang), names)
    for lang in LANGS:
        repair_loop(log, bundle, "guarded", lang, raw_entries.get(lang), names, pipeline)
    for lang in LANGS:
        repair_loop(log, bundle, "delex", lang, delex_entries.get(lang), names, pipeline)


def process_delex_context_bundle(bundle: dict[str, Any], log: RunLog, pipeline: ModuleType) -> None:
    """Follow-up condition: delexicalised placeholders with the product names shown in the prompt."""
    names = (pipeline.display_name(bundle["anchor"]), pipeline.display_name(bundle["pull"]))
    prompt = delex_context_prompt(names[0], names[1], bundle["pct"])
    first = log.call(bundle["id"], "delex_context", "all", 0, prompt, common.SEED, FIRST_PREDICT)
    entries = parse_all_languages(first["raw"]) if first else {}
    for lang in LANGS:
        repair_loop(log, bundle, "delex_context", lang, entries.get(lang), names, pipeline)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pipeline", help="pipeline file (default: frozen copy, else working copy)")
    parser.add_argument("--n-per-set", type=int, default=10)
    parser.add_argument("--retry-errors", action="store_true", help="make failed calls again")
    parser.add_argument("--dev-smoke", action="store_true", help="one dev bundle, logged under results/smoke/")
    parser.add_argument("--max-calls", type=int, help="stop making new calls after this many (for the smoke test)")
    parser.add_argument("--delex-context", action="store_true",
                        help="run the follow-up delex_context condition (writes copy_runs_delex_context.jsonl)")
    args = parser.parse_args()

    set_ids = ["dev"] if args.dev_smoke else list(common.TEST_SETS)
    pipeline = cl_features.load_pipeline(common.resolve_pipeline_path(args.pipeline, set_ids))
    if args.dev_smoke:
        suffix = "_delex_context" if args.delex_context else ""
        bundles = dev_smoke_bundle(pipeline)
        out = common.results_dir() / "smoke" / f"copy_runs{suffix}_dev_smoke.jsonl"
    elif args.delex_context:
        bundles = select_bundles(args.n_per_set)
        out = common.results_dir() / "copy_runs_delex_context.jsonl"
    else:
        bundles, out = select_bundles(args.n_per_set), runs_path()
        bundles_path().parent.mkdir(parents=True, exist_ok=True)
        bundles_path().write_text(json.dumps(bundles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log = RunLog(out, retry_errors=args.retry_errors, max_calls=args.max_calls)
    runner = process_delex_context_bundle if args.delex_context else process_bundle
    for number, bundle in enumerate(bundles, start=1):
        made = log.calls_made
        runner(bundle, log, pipeline)
        print(f"[copy] bundle {number}/{len(bundles)} done, {log.calls_made - made} new calls", flush=True)
    print(f"[copy] finished: {log.calls_made} new calls in this run, log at {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
