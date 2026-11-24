#!/usr/bin/env python3
"""Deterministic checks on flyer copy, one function per check id of SCHEMA.md.

run_checks() returns {check_id: {"passed": bool, "detail": str, ...}} for the checks that
apply to the language (address is German only, oov is Greek only). Product names, or the
[A]/[B] placeholders of the delexicalised condition, are masked before any check that is
about the words the model chose itself.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

from wordfreq import top_n_list, zipf_frequency

CHECK_IDS = ("names", "number", "sentence", "length", "language", "address", "claim", "oov", "parse")
PLACEHOLDERS = ("[A]", "[B]")
MAX_TITLE_WORDS = 6
MAX_COPY_WORDS = 25
SPACY_MODELS = {"de": "de_core_news_sm", "en": "en_core_web_sm", "el": "el_core_news_sm"}
MASK_WORD = {"de": "Produkt", "en": "product", "el": "προϊόν"}

# ------------------------------------------------------------------------- helpers


def _normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _name_pattern(name: str) -> re.Pattern[str]:
    """A product name as written: case-insensitive, any whitespace, not inside a longer word."""
    parts = [re.escape(p) for p in _normalise(name).split()]
    body = r"\s+".join(parts).replace("ß", "(?:ß|ss)")
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _mask(text: str, names: tuple[str, ...], replacement: str = " ") -> str:
    """Remove the names from a text, the longest first, because one can contain the other."""
    out = _normalise(text)
    for name in sorted(names, key=len, reverse=True):
        out = out.replace(name, replacement) if name in PLACEHOLDERS else _name_pattern(name).sub(replacement, out)
    return out


def _words(text: str) -> list[str]:
    return [w for w in text.split() if any(ch.isalnum() for ch in w)]


def _strip_marks(text: str) -> str:
    """Lower case without accents; final sigma becomes σ, so forms compare in any case."""
    bare = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
    return bare.lower().replace("ς", "σ")


def _ok(detail: str = "", **extra: Any) -> dict[str, Any]:
    return {"passed": True, "detail": detail, **extra}


def _fail(detail: str, **extra: Any) -> dict[str, Any]:
    return {"passed": False, "detail": detail, **extra}


@lru_cache(maxsize=None)
def _nlp(lang: str) -> Any:
    import spacy
    return spacy.load(SPACY_MODELS[lang], exclude=["ner", "lemmatizer"])


# ------------------------------------------------------------------------- names


def check_names(copy: str, names: tuple[str, ...]) -> dict[str, Any]:
    missing = [n for n in names if not _name_pattern(n).search(_normalise(copy))]
    if missing:
        return _fail("missing verbatim: " + "; ".join(missing), missing=missing)
    return _ok()


def check_placeholders(title: str, copy: str) -> dict[str, Any]:
    """Delexicalised version of names: each placeholder exactly once in the copy, none in the title."""
    problems = [f"{p} appears {copy.count(p)} times in the copy" for p in PLACEHOLDERS if copy.count(p) != 1]
    problems += [f"{p} is in the title" for p in PLACEHOLDERS if p in title]
    return _fail("; ".join(problems)) if problems else _ok()


# ------------------------------------------------------------------------- number


def check_number(title: str, copy: str, names: tuple[str, ...], pct: int) -> dict[str, Any]:
    copy_digits = re.findall(r"\d+", _mask(copy, names))
    title_digits = re.findall(r"\d+", _mask(title, names))
    invented = sorted({d for d in copy_digits + title_digits if d != str(pct)})
    problems = [] if str(pct) in copy_digits else [f"discount {pct} is not in the copy"]
    if invented:
        problems.append("other numbers outside the product names: " + ", ".join(invented))
    return _fail("; ".join(problems), invented=invented) if problems else _ok()


# ------------------------------------------------------------------------- sentence

_ABBREVIATIONS = {
    "de": ("z. B.", "z.B.", "ca.", "inkl.", "evtl.", "bzw.", "usw.", "ggf.", "max.", "min.", "Nr.", "etc."),
    "en": ("e.g.", "i.e.", "approx.", "incl.", "etc.", "No.", "vs."),
    "el": ("π.χ.", "κ.λπ.", "κτλ.", "δηλ.", "αρ."),
}
_HARD_END = set(".!?")
_SOFT_END = {"…", "..."}


def sentence_count(copy: str, names: tuple[str, ...], lang: str) -> int:
    """Number of sentences. The spaCy tokenizer keeps abbreviations and decimals whole.

    The parser alone is not used as the judge: de_core_news_sm splits the valid imperative
    "Kaufe | X und bekomme Y mit 20% Rabatt dazu!" after its verb. So a break needs end
    punctuation: . ! ? (and the Greek question mark ;) always break when words follow;
    an ellipsis breaks only where the parser also starts a sentence.
    """
    extra = len(re.findall(r"\S\s*\n\s*\S", copy))
    text = _mask(copy, names, MASK_WORD[lang])
    for abbreviation in _ABBREVIATIONS[lang]:
        text = re.sub(rf"(?<!\w){re.escape(abbreviation)}", abbreviation.replace(".", "").replace(" ", ""),
                      text, flags=re.IGNORECASE)
    doc = _nlp(lang)(text)
    hard = _HARD_END | ({";", ";"} if lang == "el" else set())
    breaks = 0
    for i, token in enumerate(doc[:-1]):
        if not any(t.is_alpha or t.like_num for t in doc[i + 1:]):
            break
        if token.text in hard or (token.text in _SOFT_END and doc[i + 1].is_sent_start):
            breaks += 1
    return 1 + breaks + extra


def check_sentence(copy: str, names: tuple[str, ...], lang: str) -> dict[str, Any]:
    n = sentence_count(copy, names, lang)
    return _ok() if n == 1 else _fail(f"{n} sentences", sentences=n)


# ------------------------------------------------------------------------- length


def check_length(title: str, copy: str, names: tuple[str, ...]) -> dict[str, Any]:
    n_title, n_copy = len(_words(_normalise(title))), len(_words(_mask(copy, names)))
    problems = []
    if n_title > MAX_TITLE_WORDS:
        problems.append(f"title has {n_title} words (limit {MAX_TITLE_WORDS})")
    if n_copy > MAX_COPY_WORDS:
        problems.append(f"copy has {n_copy} words without the product names (limit {MAX_COPY_WORDS})")
    extra = {"title_words": n_title, "copy_words": n_copy}
    return _fail("; ".join(problems), **extra) if problems else _ok(**extra)


# ------------------------------------------------------------------------- language

FUNCTION_WORDS = {
    "de": {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "und", "oder",
           "mit", "auf", "für", "zu", "zum", "zur", "von", "bei", "beim", "aus", "im", "dir", "dich", "du",
           "dein", "deine", "deinen", "deinem", "deiner", "ist", "nicht", "wenn", "jetzt", "nur", "noch"},
    "en": {"the", "and", "or", "with", "for", "of", "off", "on", "to", "at", "from", "your", "you", "a",
           "this", "these", "when", "is", "not", "now", "only", "get", "buy"},
}


def _letter_script(ch: str) -> str:
    code = ord(ch)
    if code < 0x250 or 0x1E00 <= code <= 0x1EFF:
        return "latin"
    if 0x370 <= code <= 0x3FF or 0x1F00 <= code <= 0x1FFF:
        return "greek"
    return "other"


def _script_counts(text: str) -> dict[str, int]:
    counts = {"latin": 0, "greek": 0, "other": 0}
    for ch in text:
        if ch.isalpha():
            counts[_letter_script(ch)] += 1
    return counts


def _function_word_share(text: str, lang: str) -> tuple[float | None, int, int]:
    """Share of this language's function words among those of DE and EN; None if there are none."""
    tokens = [w.strip(".,;:!?()«»„“”\"'–—-").lower() for w in text.split()]
    other = "en" if lang == "de" else "de"
    own, foreign = (sum(t in FUNCTION_WORDS[key] for t in tokens) for key in (lang, other))
    return (own / (own + foreign) if own + foreign else None), own, foreign


def check_language(title: str, copy: str, names: tuple[str, ...], lang: str) -> dict[str, Any]:
    masked = {"title": _mask(title, names), "copy": _mask(copy, names)}
    joined = " ".join(masked.values())
    counts = _script_counts(joined)
    if counts["other"]:
        foreign = sorted({c for c in joined if c.isalpha() and _letter_script(c) == "other"})
        return _fail("letters from another script: " + "".join(foreign), script_counts=counts)
    for part, text in masked.items():
        if lang == "el":
            letters = _script_counts(text)
            total = letters["latin"] + letters["greek"]
            if total and letters["greek"] / total <= 0.5:
                return _fail(f"{part}: Greek is only {letters['greek']} of {total} letters", script_counts=counts)
        else:
            if _script_counts(text)["greek"]:
                return _fail(f"{part}: Greek letters in {lang} text", script_counts=counts)
            share, own, foreign = _function_word_share(text, lang)
            if share is not None and foreign > own:
                return _fail(f"{part}: function words look like the other language "
                             f"({own} {lang} vs {foreign} other)", script_counts=counts)
    if not (counts["greek"] if lang == "el" else counts["latin"]):
        return _fail("no words of its own outside the product names", script_counts=counts)
    return _ok(script_counts=counts)


# ------------------------------------------------------------------------- address

_POLITE_RE = re.compile(r"(?<!\w)(?:Sie|SIE|Ihnen|IHNEN|Ihr(?:e|en|em|er|es)?|IHR(?:E|EN|EM|ER|ES)?)(?!\w)")


def check_address(title: str, copy: str, names: tuple[str, ...]) -> dict[str, Any]:
    """German only: the polite Sie forms fail; imperatives and du forms pass."""
    found = sorted(set(_POLITE_RE.findall(_mask(f"{title} . {copy}", names))))
    return _fail("polite address: " + ", ".join(found)) if found else _ok()


# ------------------------------------------------------------------------- claim

# Promises the flyer cannot back: the only offer is the discount. "*" allows endings.
CLAIMS = {
    "de": ("gratis", "kostenlos*", "umsonst", "geschenkt", "nur heute", "heute", "begrenzt*", "limitiert*",
           "garantiert*", "exklusiv*", "solange der vorrat reicht", "letzte chance"),
    "en": ("free", "today only", "today", "tonight", "limited", "guaranteed", "exclusive*", "while stocks last",
           "last chance", "lowest price"),
    "el": ("δωρεάν", "μόνο σήμερα", "σήμερα", "απόψε", "περιορισμέν*", "εγγυημέν*", "αποκλειστικ*",
           "τελευταία ευκαιρία"),
}


def _claim_regexes() -> list[tuple[str, re.Pattern[str]]]:
    out = []
    for terms in CLAIMS.values():
        for term in terms:
            stem = _strip_marks(term.rstrip("*"))
            tail = r"\w*" if term.endswith("*") else ""
            out.append((term, re.compile(rf"(?<!\w){re.escape(stem)}{tail}(?!\w)")))
    return out


_CLAIM_REGEXES = _claim_regexes()


def check_claim(title: str, copy: str, names: tuple[str, ...]) -> dict[str, Any]:
    """All three lexicons apply to every language: "free" in a German text is still a claim."""
    text = _strip_marks(_mask(f"{title} . {copy}", names))
    found = [term for term, regex in _CLAIM_REGEXES if regex.search(text)]
    return _fail("unsupported claim: " + ", ".join(found), claims=found) if found else _ok()


# ------------------------------------------------------------------------- oov

_GREEK_WORD_RE = re.compile(r"[Ͱ-Ͽἀ-῿]+")
# Common Modern Greek inflection endings, without accents and with σ for ς.
_GREEK_ENDINGS = tuple(sorted({
    "ος", "ου", "ο", "ε", "οι", "ων", "ους", "α", "ες", "ας", "η", "ης", "εις", "εων", "ι", "ια", "ιων", "ιες", "ιας",
    "μα", "ματα", "ματων", "ματος", "ση", "σης", "σεις", "σεων", "ω", "ει", "ουμε", "ετε", "ουν", "αω", "αει", "αμε",
    "ατε", "ανε", "ουσα", "στε", "τε", "σε", "ξε", "ψε", "ξτε", "ψτε", "σουμε", "σετε", "σουν", "ομαι", "εσαι",
    "εται", "ομαστε", "εστε", "ονται", "ότερος", "ότερη", "ότερο", "ότερα", "ότερες", "ότεροι", "ότερων",
}, key=len, reverse=True))
_GREEK_ENDINGS_BARE = tuple(_strip_marks(e) for e in _GREEK_ENDINGS)
_MIN_GREEK_STEM = 4
# False = plain list lookup only. strict_oov_words always records what that would flag.
OOV_STEM_BACKOFF = True


@lru_cache(maxsize=1)
def _greek_index() -> frozenset[str]:
    return frozenset(_strip_marks(w) for w in top_n_list("el", 10_000_000) if _GREEK_WORD_RE.fullmatch(w))


def _in_frequency_list(word: str) -> bool:
    """zipf > 0 for the form itself, its lower case, or its accent-free form."""
    lower = word.lower()
    return zipf_frequency(word, "el") > 0 or zipf_frequency(lower, "el") > 0 or _strip_marks(word) in _greek_index()


def _known_by_stem(word: str) -> bool:
    """Another inflected form of the same stem is listed.

    wordfreq only has its small Greek list (47k forms), which misses correct rarer forms
    of common words, such as εξοικονομήστε for εξοικονόμηση. A stem of 4+ letters with a
    known ending accepts those, and still rejects non-words such as Χαρτοπετρώτες.
    """
    bare, index = _strip_marks(word), _greek_index()
    for ending in _GREEK_ENDINGS_BARE:
        stem = bare[: -len(ending)]
        if bare.endswith(ending) and len(stem) >= _MIN_GREEK_STEM:
            if any(stem + other in index for other in _GREEK_ENDINGS_BARE):
                return True
    return False


def check_oov(title: str, copy: str, names: tuple[str, ...]) -> dict[str, Any]:
    words = _GREEK_WORD_RE.findall(_mask(f"{title} {copy}", names))
    # One- and two-letter runs are function words or abbreviation pieces (κ.λπ.), not inventions.
    strict = list(dict.fromkeys(w for w in words if len(w) > 2 and not _in_frequency_list(w)))
    oov = [w for w in strict if not (OOV_STEM_BACKOFF and _known_by_stem(w))]
    extra = {"oov_words": oov, "strict_oov_words": strict}
    return _fail("not in the Greek frequency list: " + ", ".join(oov), **extra) if oov else _ok(**extra)


# ------------------------------------------------------------------------- all


def run_checks(lang: str, title: str, copy: str, names: tuple[str, ...], pct: int,
               delex: bool = False) -> dict[str, dict[str, Any]]:
    """All checks for one text. In delex mode names is ignored and [A]/[B] are the names."""
    if not title.strip() or not copy.strip():
        return {"parse": _fail("title or copy is empty or could not be parsed")}
    shown = PLACEHOLDERS if delex else tuple(names)
    results = {
        "parse": _ok(),
        "names": check_placeholders(title, copy) if delex else check_names(copy, shown),
        "number": check_number(title, copy, shown, pct),
        "sentence": check_sentence(copy, shown, lang),
        "length": check_length(title, copy, shown),
        "language": check_language(title, copy, shown, lang),
        "claim": check_claim(title, copy, shown),
    }
    if lang == "de":
        results["address"] = check_address(title, copy, shown)
    if lang == "el":
        results["oov"] = check_oov(title, copy, shown)
    return results


def failed_ids(results: dict[str, dict[str, Any]]) -> list[str]:
    """Failed check ids in the fixed order of CHECK_IDS."""
    return [c for c in CHECK_IDS if c in results and not results[c]["passed"]]


def oov_words(results: dict[str, dict[str, Any]]) -> list[str]:
    return list(results.get("oov", {}).get("oov_words", []))
