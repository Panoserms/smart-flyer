#!/usr/bin/env python3
"""The computational-linguistics stage of the study, as the pipeline itself computes it.

analyse_pairs() returns the "cl" block of SCHEMA.md per pair, plus cl_reject. All logic
comes from the pipeline module it is given, so the frozen copy decides the test results.

  python experiments/cl_features.py --set dev              # CPU timing of one set
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

_NLP_CACHE: dict[str, Any] = {}


def load_pipeline(path: str | Path) -> ModuleType:
    """Import a pipeline file under a name unique to its path, so two copies can coexist."""
    path = Path(path).resolve()
    name = f"smart_flyer_pipeline_{hashlib.md5(str(path).encode()).hexdigest()[:10]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import a pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while the file executes
    spec.loader.exec_module(module)
    return module


def _head_pos(doc: Any, head_noun: str, lang: str, pipeline: ModuleType) -> str:
    """POS of the token the pipeline's _from_doc chose. HeadInfo keeps only its lemma,

    so the token is found again through the pipeline's own lemma helpers.
    """
    tokens = [t for t in doc if not t.is_punct]
    if lang != "en" and doc.text.startswith(pipeline.GERMAN_FRAME + " "):
        tokens = tokens[len(pipeline.GERMAN_FRAME.split()):]

    def lemma_of(token: Any) -> str:
        if lang != "en":
            return pipeline.german_lemma(token.text, token.lemma_ or token.text)
        lemma = token.lemma_.lower() if token.pos_ == "NOUN" and token.lemma_ else token.text.lower()
        return pipeline._singularise_en(lemma) if lemma == token.text.lower() else lemma

    matches = [t for t in tokens if lemma_of(t) == head_noun]
    nouns = [t for t in matches if t.pos_ in ("NOUN", "PROPN")]
    pool = nouns or matches
    if not pool:
        return "X"
    return pool[-1].pos_ if lang == "en" else pool[0].pos_


def _make_extractor(pipeline: ModuleType, learned: frozenset[str], langs: set[str]) -> Any:
    """A pipeline extractor that also keeps the POS of each chosen head token."""

    class RecordingExtractor(pipeline.HeadNounExtractor):
        head_pos_by_text: dict[tuple[str, str], str] = {}

        def _from_doc(self, doc: Any, lang: str) -> Any:
            info = super()._from_doc(doc, lang)
            self.head_pos_by_text[(doc.text, lang)] = _head_pos(doc, info.head_noun, lang, pipeline)
            return info

    extractor = RecordingExtractor(learned)
    extractor.head_pos_by_text = {}
    extractor._nlp.update({k: v for k, v in _NLP_CACHE.items() if k in langs})
    extractor.load(langs)
    _NLP_CACHE.update(extractor._nlp)
    return extractor


def _parsed_text(name: str, lang: str, learned: frozenset[str], pipeline: ModuleType) -> str:
    text = pipeline.normalise_for_nlp(name, learned if lang == "en" else frozenset())
    return text if lang == "en" else f"{pipeline.GERMAN_FRAME} {text}."


def stripped_words(name: str, lang: str, learned: frozenset[str], pipeline: ModuleType) -> list[str]:
    """Attribute words normalise_for_nlp removed (not counts, sizes or other number tokens)."""
    used = learned if lang == "en" else frozenset()
    bare = lambda tok: tok.lower().strip(".,;:()'\"")  # noqa: E731
    kept = Counter(bare(t) for t in pipeline.normalise_for_nlp(name, used).split())
    out = []
    for tok in pipeline._COUNT_PHRASE_RE.sub(" ", name).split():
        word = bare(tok)
        if kept[word] > 0:
            kept[word] -= 1
        elif word not in pipeline.UNIT_TOKENS and any(c.isalpha() for c in tok) \
                and not any(c.isdigit() for c in tok):
            out.append(word)
    return out


def _real_stem(suffix: str) -> str:
    """Display only: the longest real German word in a shared ending (twurst -> wurst). No decision depends on it."""
    from wordfreq import zipf_frequency

    for start in range(len(suffix) - 2):
        if zipf_frequency(suffix[start:], "de") >= 2.5:
            return suffix[start:]
    return suffix


def german_compound(head_a: str, head_b: str, pipeline: ModuleType) -> str:
    """The shared suffix or lexicon base behind head_collision(a, b, german=True)."""
    a, b = head_a.lower(), head_b.lower()
    suffix = pipeline.common_suffix(a, b)
    if len(suffix) >= pipeline.MIN_COMPOUND_SUFFIX:
        return f"-{_real_stem(suffix)}"
    for base in pipeline.SHORT_COMPOUND_BASES:
        if a.endswith(base) and b.endswith(base):
            return f"-{base}"
    return f"-{suffix}"


def analyse_pairs(pairs: list[dict[str, Any]], catalog: list[str] | None,
                  pipeline: ModuleType) -> dict[str, dict[str, Any]]:
    """pair_id -> SCHEMA "cl" block + cl_reject, mirroring run_pruner with its bag guard on."""
    learned = pipeline.learn_variant_attributes(catalog) if catalog else frozenset()
    langs = {p["lang_a"] for p in pairs} | {p["lang_b"] for p in pairs}
    extractor = _make_extractor(pipeline, learned, langs)
    extractor.prepare([(p["a"], p["lang_a"]) for p in pairs] + [(p["b"], p["lang_b"]) for p in pairs])
    return {p["pair_id"]: _analyse_one(p, extractor, pipeline) for p in pairs}


def _analyse_one(pair: dict[str, Any], extractor: Any, pipeline: ModuleType) -> dict[str, Any]:
    a, b, lang_a, lang_b = pair["a"], pair["b"], pair["lang_a"], pair["lang_b"]
    learned = extractor.learned_variants
    info_a, info_b = extractor.get(a, lang_a), extractor.get(b, lang_b)
    stop = pipeline.stop_item_match(a, True) or pipeline.stop_item_match(b, True)
    german = "de" in (lang_a, lang_b)
    same_head = pipeline.head_collision(info_a.head_noun, info_b.head_noun, False)
    shared = pipeline.head_collision(info_a.head_noun, info_b.head_noun, german)
    if german:
        if shared:
            compound = german_compound(info_a.head_noun, info_b.head_noun, pipeline) if not same_head else None
        else:
            compound = pipeline.shared_name_words(a, b)
    else:
        compound = None if shared else pipeline.open_compound_match(a, b, learned)
    pos = extractor.head_pos_by_text
    return {
        "stop_term": stop,
        "same_head": bool(same_head),
        "compound": compound,
        "head_a": info_a.display,
        "head_b": info_b.display,
        "pos_a": pos.get((_parsed_text(a, lang_a, learned, pipeline), lang_a), "X"),
        "pos_b": pos.get((_parsed_text(b, lang_b, learned, pipeline), lang_b), "X"),
        "stripped_a": stripped_words(a, lang_a, learned, pipeline),
        "stripped_b": stripped_words(b, lang_b, learned, pipeline),
        "cl_reject": bool(stop) or bool(shared) or bool(compound),
    }


def timed_analysis(pairs: list[dict[str, Any]], catalog: list[str] | None,
                   pipeline: ModuleType) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """analyse_pairs() plus its CPU cost. Model loading happens first and is not counted."""
    langs = {p["lang_a"] for p in pairs} | {p["lang_b"] for p in pairs}
    _make_extractor(pipeline, frozenset(), langs)
    wall, cpu = time.perf_counter(), time.process_time()
    analysis = analyse_pairs(pairs, catalog, pipeline)
    timing = {"n_pairs": len(pairs), "wall_ms": round((time.perf_counter() - wall) * 1000, 1),
              "cpu_ms": round((time.process_time() - cpu) * 1000, 1)}
    return analysis, timing


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU cost of the linguistic analysis of one set.")
    parser.add_argument("--set", required=True, dest="set_id")
    parser.add_argument("--pipeline", help="pipeline file (default: frozen copy, else working copy)")
    args = parser.parse_args()
    path = common.resolve_pipeline_path(args.pipeline, [args.set_id])
    pairs = common.read_pairs(args.set_id)
    analysis, timing = timed_analysis(pairs, common.read_catalog(args.set_id), load_pipeline(path))
    timing["rejected"] = sum(a["cl_reject"] for a in analysis.values())
    print(json.dumps({"set": args.set_id, **timing}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
