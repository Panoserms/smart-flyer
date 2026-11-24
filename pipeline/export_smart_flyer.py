#!/usr/bin/env python3
"""
Smart Flyer: Market Basket Discount Generator with Linguistic Pruning.

Pipeline
  1. Association rules. Either UCI Online Retail (--csv, .csv or .xlsx) mined with
     mlxtend fpgrowth + association_rules, or a built-in 16-rule sample when no file
     is given, so the script runs out of the box.
  2. Filter 1, regex: packaging and operational stop items (bags, postage, Pfand ...).
  3. Filter 2, spaCy: syntactic head noun of each product name, plus a compound
     suffix check (Vollmilch / Hafermilch). Shared head = substitute variant.
  4. Filter 3, gate: the bundle must grow the basket (promo price >= 110 % of the
     anchor price) and the rule's lift must reach a floor.
  4b. LLM judge (default on when Ollama runs): the surviving pairs go to Gemma with the
     parser's analysis in the prompt. Only the verdict COMPLEMENT stays approved
     (study condition CLf, prompt variant "examples"). --no-llm-judge skips it.
  5. Copy, "guarded" method of the study: Gemma writes DE / EN (EL with --gemma-greek)
     copy for the top surviving bundles; pipeline/copy_checks.py checks it, a failing
     language gets targeted repair (at most 2 retries), then a fixed template. Greek
     uses templates by default.

Output: smart_flyer_data.json, the contract shared with
src/components/SmartFlyerShowcase.tsx.

Usage
  python pipeline/export_smart_flyer.py                         # built-in sample
  python pipeline/export_smart_flyer.py --no-llm                # templates only, ~2 s
  python pipeline/export_smart_flyer.py --csv data/raw/OnlineRetail.csv --top-n 12
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import copy_checks as checks  # noqa: E402

log = logging.getLogger("smart_flyer")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "public" / "smart_flyer_data.json"      # a --csv run: the data the showcase calls "UCI Online Retail"
SAMPLE_OUT = REPO_ROOT / "out" / "smart_flyer_sample_run.json"    # a built-in sample run: never over the shipped files

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOP_ITEM_REGEX = re.compile(
    r"\b(bag|tragetasche|pfand|deposit|postage|porto|manual|dotcom|adjust|gift card|gutschein|carrier)\b",
    re.IGNORECASE,
)

# On UCI Online Retail, the bare "bag" alternative fired on 160 of 161 rejected pairs,
# all of them bags the shop SELLS (JUMBO BAG RED RETROSPOT, LUNCH BAG SUKI DESIGN).
# The guard counts "bag" as packaging only after a packaging modifier or on its own.
# --strict-spec-regex turns the guard off.
PACKAGING_BAG_REGEX = re.compile(
    r"\b(?:plastic|paper|carrier|shopping|refuse|bin)\s+bags?\b|^\W*bags?\W*$", re.IGNORECASE)

# German writes these as closed compounds, so \bpfand\b misses "Flaschenpfand".
GERMAN_STOP_COMPOUND_REGEX = re.compile(r"\w{3,}?(pfand|tragetasche|porto|gutschein)\b", re.IGNORECASE)

# Words about the pack or cut rather than the product; a shared one says nothing about the product.
PACK_WORDS = {"scheiben", "dose", "flasche", "glas", "beutel", "packung", "becher", "stück", "stueck"}

# Measurement units stripped before head-noun extraction (tokens with digits go too).
UNIT_TOKENS = {"g", "kg", "ml", "l", "pack", "stück", "stk", "oz", "cm"}
_COUNT_PHRASE_RE = re.compile(r"\b(?:pack|set)\s+of\s*\d+\b|\b(?:no|nr)\.?\s*\d+\b", re.IGNORECASE)

# Variant attributes: words that name a colour, size, pattern, collection or
# marketing label rather than the kind of product. Catalog names put them where a
# parser expects the head ("ALARM CLOCK BAKELIKE GREEN" -> head "Green"), so they
# are removed before parsing.
VARIANT_ATTRIBUTES = {
    # colours and sizes (EN)
    "red", "green", "blue", "pink", "white", "black", "ivory", "yellow", "purple",
    "orange", "gold", "silver", "cream", "grey", "gray", "turquoise", "brown",
    "lilac", "peach", "aqua", "mint", "assorted", "colour", "colours", "small",
    "large", "mini", "medium", "big", "jumbo",
    # patterns and collections (UCI Online Retail)
    "design", "vintage", "retro", "retrospot", "polkadot", "spotty", "dotty",
    "paisley", "regency", "woodland", "pantry", "christmas", "bakelike", "spaceboy",
    "dolly", "girl", "skulls", "strawberry", "union", "jack", "rose", "roses",
    # marketing labels (DE grocery)
    "bio", "barista", "classic", "original", "premium", "extra", "ungesüßt",
}
# German catalog nouns given with an inflected base: Cherrytomaten -> Cherrytomate.
GERMAN_FRAME = "Ich kaufe"
GERMAN_LEMMA_BASES = ("tomate", "bohne", "banane", "nudel", "kartoffel", "zwiebel",
                      "gurke", "orange", "zitrone", "erdbeere", "möhre", "karotte")
# Pre-nominal German participles ("Passierte Tomaten", "Gehackte Mandeln") are often
# tagged NOUN by de_core_news_sm when capitalised; skip them when a real noun follows.
_GERMAN_PARTICIPLE_RE = re.compile(
    r"^(?:[a-zäöüß]+ierte[nrs]?|ge[a-zäöüß]{3,}(?:te[nrs]?|ene[nrs]?))$",
    re.IGNORECASE,
)

# Compound heads: the spec rule is "last 5+ characters match". German grocery
# compounds also end in common 2–4 letter bases (Apfelsaft / Orangensaft) that the
# 5-character rule misses, so a small lexicon of short bases is checked as well.
MIN_COMPOUND_SUFFIX = 5
SHORT_COMPOUND_BASES = ("saft", "wein", "bier", "brot", "käse", "reis", "mehl", "tee", "öl", "mus", "eis", "salz")
# Longer bases, used only to label an item's pos_pattern as a compound.
COMPOUND_BASES = SHORT_COMPOUND_BASES + (
    "milch", "joghurt", "bohne", "drink", "wurst", "kohle", "tomate", "sahne",
    "butter", "schokolade", "wasser", "kaffee", "nudel", "quark", "schinken",
)

DISCOUNT_PCT = 20
MIN_BASKET_EXPANSION = 1.10   # promo_bundle_price >= anchor_price * 1.10
DEFAULT_MIN_BUNDLE_LIFT = 2.0

SUBSTITUTE_EXPLANATION = "Substitute Variant: Shared syntactic/compound head"

OLLAMA_URL = "http://localhost:11434"
PREFERRED_MODELS = ("gemma4:e2b-it-qat", "gemma2:2b", "gemma:2b")
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
# Letters outside Latin / Greek (gemma4 once emitted Thai inside Greek copy).
FOREIGN_SCRIPT_RE = re.compile(r"[^\W\d_A-Za-z\u00C0-\u024F\u0370-\u03FF\u1F00-\u1FFF]")

STATUS_APPROVED = "approved_bundle"
STATUS_PACKAGING = "rejected_packaging_noise"
STATUS_HEAD = "rejected_same_head_noun"
STATUS_MARGIN = "rejected_low_margin"
STATUS_JUDGE = "rejected_llm_judge"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """One 1→1 association rule before pruning."""
    antecedent: str
    consequent: str
    price_a: float
    price_c: float
    support: float
    confidence: float
    lift: float
    domain: str          # "supermarket_grocery" | "uci_retail"
    lang_a: str = "en"   # spaCy model per item: "en" | "de"
    lang_c: str = "en"


@dataclass
class HeadInfo:
    head_noun: str       # normalised lemma, lower case (used for comparison)
    display: str         # capitalised lemma for the UI
    pos_pattern: str


# Built-in sample. Item pairs are real (UCI Online Retail) or realistic (German
# grocery); support / confidence / lift / prices are ILLUSTRATIVE, not measured.
# UCI prices are GBP list prices converted at ~1.16 EUR/GBP.
BUILTIN_RULES: list[Candidate] = [
    # --- UCI Online Retail ---------------------------------------------------
    Candidate("WHITE HANGING HEART T-LIGHT HOLDER", "RED HANGING HEART T-LIGHT HOLDER",
              3.42, 3.42, 0.021, 0.52, 8.90, "uci_retail"),
    Candidate("ALARM CLOCK BAKELIKE GREEN", "ALARM CLOCK BAKELIKE RED",
              4.35, 4.35, 0.022, 0.64, 16.30, "uci_retail"),
    Candidate("ROSES REGENCY TEACUP AND SAUCER", "GREEN REGENCY TEACUP AND SAUCER",
              3.42, 3.42, 0.031, 0.71, 15.80, "uci_retail"),
    Candidate("POSTAGE", "ROUND SNACK BOXES SET OF4 WOODLAND",
              20.88, 3.42, 0.019, 0.33, 5.10, "uci_retail"),
    Candidate("REGENCY CAKESTAND 3 TIER", "ROSES REGENCY TEACUP AND SAUCER",
              14.79, 3.42, 0.018, 0.29, 3.70, "uci_retail"),
    Candidate("SET OF 3 CAKE TINS PANTRY DESIGN", "PACK OF 72 RETROSPOT CAKE CASES",
              5.74, 0.64, 0.016, 0.31, 3.20, "uci_retail"),
    Candidate("JAM MAKING SET WITH JARS", "RECIPE BOX PANTRY YELLOW DESIGN",
              4.93, 3.42, 0.017, 0.36, 5.40, "uci_retail"),
    # --- German / English supermarket grocery --------------------------------
    Candidate("Bio Espressobohnen Ganze Bohne 1kg", "Barista Haferdrink Ungesüßt 1L",
              12.99, 2.19, 0.034, 0.48, 4.15, "supermarket_grocery", "de", "de"),
    Candidate("Bio Vollmilch 3,8% 1L", "Bio Fettarme Milch 1,5% 1L",
              1.29, 1.19, 0.041, 0.38, 3.10, "supermarket_grocery", "de", "de"),
    Candidate("Erdbeerjoghurt 500g", "Vanillejoghurt 500g",
              0.89, 0.89, 0.027, 0.44, 4.60, "supermarket_grocery", "de", "de"),
    Candidate("Grillkohle Buchenholz 3kg", "Thüringer Rostbratwurst 6 Stück",
              5.99, 3.49, 0.022, 0.51, 6.80, "supermarket_grocery", "de", "de"),
    Candidate("Plastic Carrier Bag", "Bio Bananen 1kg",
              0.30, 1.49, 0.088, 0.29, 1.90, "supermarket_grocery", "en", "de"),
    Candidate("Pfand Einwegflasche 0,25 €", "Mineralwasser Classic 1,5L",
              0.25, 0.49, 0.061, 0.72, 2.40, "supermarket_grocery", "de", "de"),
    Candidate("Spaghetti No. 5 500g", "Tomatensauce Basilikum 400g",
              1.29, 0.99, 0.029, 0.34, 2.60, "supermarket_grocery", "de", "de"),
    Candidate("Toastbrot 500g", "Deutsche Markenbutter 250g",
              1.19, 2.29, 0.044, 0.21, 1.60, "supermarket_grocery", "de", "de"),
    Candidate("Mozzarella 125g", "Cherrytomaten 250g",
              0.99, 1.49, 0.025, 0.39, 3.40, "supermarket_grocery", "de", "de"),
]


# ---------------------------------------------------------------------------
# Step 1: association rules from UCI Online Retail
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID",
}


def _read_table(path: Path) -> pd.DataFrame:
    dtypes = {"InvoiceNo": str, "Invoice": str, "StockCode": str}
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=dtypes)
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, dtype=dtypes, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} as UTF-8 or Latin-1")


def clean_uci(df: pd.DataFrame) -> pd.DataFrame:
    """Drop guests, credit notes, returns and near-zero prices."""
    df = df.rename(columns=_COLUMN_ALIASES)
    required = {"InvoiceNo", "Description", "Quantity", "UnitPrice", "CustomerID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["CustomerID", "Description"]).copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df = df[~df["InvoiceNo"].str.upper().str.startswith("C")]
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0.50)]
    df["Description"] = df["Description"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    return df[df["Description"] != ""]


def mine_rules(df: pd.DataFrame, min_support: float, min_lift: float) -> pd.DataFrame:
    from mlxtend.frequent_patterns import association_rules, fpgrowth

    pairs = df[["InvoiceNo", "Description"]].drop_duplicates()
    invoices = pd.Index(pairs["InvoiceNo"].unique())
    # Items below min_support cannot appear in any frequent itemset: drop them
    # before the pivot so the boolean matrix stays small on an 8 GB machine.
    counts = pairs["Description"].value_counts()
    frequent = counts[counts >= min_support * len(invoices)].index
    pairs = pairs[pairs["Description"].isin(frequent)]
    basket = pd.crosstab(pairs["InvoiceNo"], pairs["Description"]).astype(bool)
    # Re-add invoices with no frequent item, otherwise support is overstated.
    basket = basket.reindex(invoices, fill_value=False)
    log.info("Basket matrix: %d invoices x %d frequent items", *basket.shape)

    itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return pd.DataFrame()
    kwargs: dict[str, Any] = {"metric": "lift", "min_threshold": min_lift}
    if "num_itemsets" in inspect.signature(association_rules).parameters:
        kwargs["num_itemsets"] = len(basket)
    return association_rules(itemsets, **kwargs)


def load_uci_candidates(path: Path, min_support: float, min_lift: float,
                        gbp_to_eur: float) -> tuple[list[Candidate], int, list[str]]:
    df = clean_uci(_read_table(path))
    catalog = df["Description"].unique().tolist()
    n_transactions = df["InvoiceNo"].nunique()
    log.info("Cleaned UCI data: %d line items, %d invoices", len(df), n_transactions)
    rules = mine_rules(df, min_support, min_lift)
    if rules.empty:
        return [], n_transactions, catalog
    log.info("mlxtend returned %d rules (all sizes)", len(rules))

    prices = (df.groupby("Description")["UnitPrice"].median() * gbp_to_eur).round(2)
    one_to_one = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)]
    # A→B and B→A describe the same bundle; keep the stronger direction.
    best: dict[frozenset[str], Candidate] = {}
    for row in one_to_one.itertuples(index=False):
        a, c = next(iter(row.antecedents)), next(iter(row.consequents))
        cand = Candidate(a, c, float(prices[a]), float(prices[c]), float(row.support),
                         float(row.confidence), float(row.lift), "uci_retail")
        key = frozenset((a, c))
        if key not in best or cand.confidence > best[key].confidence:
            best[key] = cand
    log.info("%d distinct 1->1 item pairs", len(best))
    return list(best.values()), n_transactions, catalog


# ---------------------------------------------------------------------------
# Step 2: regex stop items
# ---------------------------------------------------------------------------

def stop_item_match(name: str, bag_guard: bool = True) -> str | None:
    """Return the stop term found in an item name, or None."""
    for m in STOP_ITEM_REGEX.finditer(name):
        if bag_guard and m.group(0).lower() == "bag" and not PACKAGING_BAG_REGEX.search(name):
            continue  # a sold bag (lunch bag, jumbo bag), not packaging
        return m.group(0)
    compound = GERMAN_STOP_COMPOUND_REGEX.search(name)
    return compound.group(1) if compound else None


# ---------------------------------------------------------------------------
# Step 3: spaCy head nouns
# ---------------------------------------------------------------------------

def smart_title(text: str) -> str:
    """Title-case ALL-CAPS receipt text so spaCy can tag it.

    str.title() turns CHILDREN'S into Children'S; capitalising each
    whitespace token gives Children's and T-light instead.
    """
    return " ".join(w[:1].upper() + w[1:].lower() for w in text.split())


def normalise_for_nlp(name: str, learned_variants: frozenset[str] = frozenset()) -> str:
    """Title-case ALL CAPS, then drop counts, units and variant attributes."""
    text = name.strip()
    if any(ch.isalpha() for ch in text) and text.upper() == text:
        text = smart_title(text)
    text = _COUNT_PHRASE_RE.sub(" ", text)
    kept: list[str] = []
    raw = text.split()
    for i, tok in enumerate(raw):
        bare = tok.lower().strip(".,;:()'\"")
        if any(ch.isdigit() for ch in tok) or not any(ch.isalpha() for ch in tok):
            continue
        if bare in UNIT_TOKENS or bare in VARIANT_ATTRIBUTES or bare in learned_variants:
            continue
        # Numeral + classifier ("Cakestand 3 Tier"): the classifier is not the head.
        # A leading numeral counts the head instead ("6 Ribbons ..."), so keep that.
        if i >= 2 and any(ch.isdigit() for ch in raw[i - 1]):
            continue
        kept.append(tok)
    if kept:
        return " ".join(kept)
    # Everything was an attribute (e.g. "Pink Polkadot"): fall back to the bare words.
    return " ".join(t for t in raw if any(ch.isalpha() for ch in t)) or text


def _catalog_tokens(name: str) -> tuple[str, ...]:
    text = _COUNT_PHRASE_RE.sub(" ", name.lower())
    return tuple(w.strip(".,;:()'\"") for w in text.split()
                 if not any(ch.isdigit() for ch in w) and w.strip(".,;:()'\""))


def learn_variant_attributes(catalog: list[str], min_pairs: int = 2) -> frozenset[str]:
    """Learn colour / pattern / collection words from minimal pairs in the catalog.

    Two descriptions that differ in exactly one slot (LUNCH BAG ALPHABET DESIGN /
    LUNCH BAG APPLE DESIGN) show that the slot's fillers are interchangeable
    variant attributes. Fillers of the FINAL slot are skipped when that is where
    they mostly occur: there the slot is the head (PINK CRYSTAL NECKLACE /
    PINK CRYSTAL BRACELET), and necklace / bracelet are product types.
    """
    slots: dict[tuple[tuple[str, ...], tuple[str, ...]], set[str]] = {}
    for name in set(catalog):
        toks = _catalog_tokens(name)
        if len(toks) < 2:
            continue
        for i in range(len(toks)):
            slots.setdefault((toks[:i], toks[i + 1:]), set()).add(toks[i])
    non_final: dict[str, int] = {}
    final: dict[str, int] = {}
    for (pre, post), fillers in slots.items():
        if len(fillers) < 2 or len(pre) + len(post) < 2:
            continue
        for f in fillers:
            target = non_final if post else final
            target[f] = target.get(f, 0) + 1
    return frozenset(w for w, n in non_final.items()
                     if n >= min_pairs and n >= final.get(w, 0) and w.isalpha() and len(w) > 2)


def open_compound_match(name_a: str, name_b: str,
                        learned_variants: frozenset[str] = frozenset()) -> str | None:
    """English counterpart of the German compound-suffix check.

    UCI catalog names open or close with a multi-word product type: HAND WARMER
    BIRD DESIGN / HAND WARMER OWL DESIGN, COOK WITH WINE METAL SIGN / GIN + TONIC
    DIET METAL SIGN, CHARLOTTE BAG DOLLY GIRL DESIGN / RED RETROSPOT CHARLOTTE BAG.
    Two kinds of shared open compound mark a variant pair:
      (a) the same leading or trailing two words at either boundary, unless both are
          attribute words (RED RETROSPOT ... / RED RETROSPOT ...);
      (b) the same three words anywhere (CHOCOLATE HOT WATER BOTTLE / HOT WATER
          BOTTLE KEEP CALM), unless any of them is an attribute (RED SPOTTY PAPER
          PLATES / RED SPOTTY PAPER CUPS are complements).
    """
    attrs = VARIANT_ATTRIBUTES | learned_variants
    a, b = _catalog_tokens(name_a), _catalog_tokens(name_b)
    if len(a) < 2 or len(b) < 2 or a == b:
        return None
    for pa, pb in ((a[:2], b[:2]), (a[-2:], b[-2:]), (a[:2], b[-2:]), (a[-2:], b[:2])):
        if pa == pb and all(t.isalpha() for t in pa) and not all(t in attrs for t in pa):
            return " ".join(pa)
    tri_b = {b[i:i + 3] for i in range(len(b) - 2)}
    for i in range(len(a) - 2):
        tri = a[i:i + 3]
        if tri in tri_b and all(t.isalpha() and t not in attrs for t in tri):
            return " ".join(tri)
    return None


def shared_name_words(name_a: str, name_b: str) -> str | None:
    """German shelf names: a content word in both names marks the same kind of product.

    Brand-like first words hide the head from the first-noun rule (Hofgut Gouda / Alpenkrone Gouda),
    and international names are open compounds (Energy Drink Zero / Energy Drink Classic).
    """
    def words(name: str) -> list[str]:
        return [t for t in _catalog_tokens(name)
                if t.isalpha() and len(t) >= 4 and t not in VARIANT_ATTRIBUTES
                and t not in UNIT_TOKENS and t not in PACK_WORDS]
    other = set(words(name_b))
    shared = [t for t in words(name_a) if t in other]
    return " ".join(shared) if shared else None


def german_lemma(word: str, spacy_lemma: str) -> str:
    low = word.lower()
    for base in GERMAN_LEMMA_BASES:
        for ending in ("n", "en", "s", "e", ""):
            if low.endswith(base + ending) if ending else low.endswith(base):
                return low[: len(low) - len(base) - len(ending)] + base
    lemma = spacy_lemma.lower()
    # The lookup table guesses on unknown compounds: Schuhcreme -> Schuhcrem.
    return low if low.endswith("e") and lemma == low[:-1] else lemma


def _singularise_en(word: str) -> str:
    """spaCy does not lemmatise PROPN, and title-cased product words are often tagged PROPN."""
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def common_suffix(a: str, b: str) -> str:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return a[len(a) - n:] if n else ""


def german_compound(head_a: str, head_b: str) -> str:
    """The shared suffix or lexicon base behind head_collision(a, b, german=True)."""
    a, b = head_a.lower(), head_b.lower()
    suffix = common_suffix(a, b)
    if len(suffix) >= MIN_COMPOUND_SUFFIX:
        return f"-{suffix}"
    for base in SHORT_COMPOUND_BASES:
        if a.endswith(base) and b.endswith(base):
            return f"-{base}"
    return f"-{suffix}"


def head_collision(a: str, b: str, german: bool) -> bool:
    """True when two head nouns name the same kind of product."""
    a, b = a.lower(), b.lower()
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 4 and longer.endswith(shorter):          # Milch / Vollmilch
        return True
    if german:
        if len(common_suffix(a, b)) >= MIN_COMPOUND_SUFFIX:     # Vollmilch / Hafermilch
            return True
        if any(a.endswith(base) and b.endswith(base) for base in SHORT_COMPOUND_BASES):
            return True                                         # Apfelsaft / Orangensaft
    # English "last 5 chars" would merge holder / folder, so it is not applied there.
    return False


class HeadNounExtractor:
    MODELS = {"en": "en_core_web_sm", "de": "de_core_news_sm"}

    def __init__(self, learned_variants: frozenset[str] = frozenset()) -> None:
        self._nlp: dict[str, Any] = {}
        self._cache: dict[tuple[str, str], HeadInfo] = {}
        self.learned_variants = learned_variants

    def load(self, langs: set[str]) -> None:
        """Load spaCy models up front so model loading is not counted as pruning time."""
        try:
            import spacy
        except ImportError:
            log.warning("spaCy is not installed; using the heuristic head-noun fallback")
            return
        for lang in sorted(langs):
            if lang in self._nlp:
                continue
            try:
                self._nlp[lang] = spacy.load(self.MODELS[lang], exclude=["ner"])
            except OSError:
                log.warning("spaCy model %s missing (python -m spacy download %s); "
                            "using the heuristic fallback for '%s'", self.MODELS[lang], self.MODELS[lang], lang)

    def prepare(self, items: list[tuple[str, str]]) -> None:
        """Parse all unique (name, lang) items in batches with nlp.pipe."""
        todo: dict[str, list[str]] = {}
        for name, lang in items:
            if (name, lang) not in self._cache:
                todo.setdefault(lang, []).append(name)
        for lang, names in todo.items():
            names = list(dict.fromkeys(names))
            texts = [normalise_for_nlp(n, self.learned_variants if lang == "en" else frozenset())
                     for n in names]
            nlp = self._nlp.get(lang)
            if nlp is not None and lang == "de":
                # Carrier sentence: on bare catalog strings de_core_news_sm tags nouns
                # as VERB/X ("Grillkohle", "Toastbrot"); as the object of "kaufe" it
                # tags them NOUN. _from_doc drops the frame tokens again.
                texts = [f"{GERMAN_FRAME} {t}." for t in texts]
            if nlp is None:
                for name, text in zip(names, texts):
                    self._cache[(name, lang)] = self._heuristic(text, lang)
                continue
            for name, doc in zip(names, nlp.pipe(texts, batch_size=256)):
                self._cache[(name, lang)] = self._from_doc(doc, lang)

    def get(self, name: str, lang: str) -> HeadInfo:
        if (name, lang) not in self._cache:
            self.prepare([(name, lang)])
        return self._cache[(name, lang)]

    @staticmethod
    def _describe(tags: list[str], display: str, lemma: str, lang: str) -> str:
        collapsed = [t for i, t in enumerate(tags) if i == 0 or t != tags[i - 1]]
        base = next((b for b in COMPOUND_BASES if lemma.endswith(b) and len(lemma) >= len(b) + 3), None)
        tail = f"(Compound Head: -{base})" if lang == "de" and base else f"(Head: {display})"
        return f"{' + '.join(collapsed) or 'X'} {tail}"

    def _from_doc(self, doc: Any, lang: str) -> HeadInfo:
        """Pick the head noun of a catalog name.

        English: product names are pre-modified NPs ("White Hanging Heart T-light
        Holder") with optional post-modifiers ("... With Jars", "... And Saucer").
        The span before the first ADP/CCONJ is the head NP; its head is the root of
        the last noun chunk in that span, or the rightmost noun (head-final).

        German: de_core_news_sm makes the first word the chunk root on catalog text
        ("Bio Fettarme Milch" -> "Bio"), and catalog names lead with the product
        ("Grillkohle Buchenholz", "Haferdrink Ungesüßt"). So the head is the first
        NOUN/PROPN, skipping adjectives ("Thüringer Rostbratwurst").
        """
        if len(doc) == 0:
            return HeadInfo("", "", "EMPTY")
        tokens = list(doc)
        if lang == "en":
            cut = next((t.i for t in tokens if t.pos_ in ("ADP", "CCONJ") and t.i > 0), len(doc))
            span = tokens[:cut]
            chunks = [c for c in doc.noun_chunks if c.end <= cut]
            nouns = [t for t in span if t.pos_ in ("NOUN", "PROPN")]
            root = chunks[-1].root if chunks else (nouns[-1] if nouns else span[-1])
        else:
            if doc.text.startswith(GERMAN_FRAME + " "):
                tokens = tokens[len(GERMAN_FRAME.split()):]
                if len(tokens) > 1 and tokens[-1].is_punct:
                    tokens = tokens[:-1]
            span = tokens
            nouns = [t for t in span if t.pos_ in ("NOUN", "PROPN")]
            non_participle = [t for t in nouns if not _GERMAN_PARTICIPLE_RE.match(t.text)]
            if non_participle:
                nouns = non_participle
            root = nouns[0] if nouns else next((t for t in span if t.is_alpha), span[0])
        tags = [t.pos_ for t in span if not t.is_punct]
        if lang == "en":
            # Only trust the lemmatiser for NOUN: as PROPN it is skipped, and when a
            # catalog noun is mis-tagged VERB it gives "bunt" for "Bunting".
            lemma = root.lemma_.lower() if root.pos_ == "NOUN" and root.lemma_ else root.text.lower()
            if lemma == root.text.lower():
                lemma = _singularise_en(lemma)
        else:
            lemma = german_lemma(root.text, root.lemma_ or root.text)
        display = lemma[:1].upper() + lemma[1:]
        return HeadInfo(lemma, display, HeadNounExtractor._describe(tags, display, lemma, lang))

    @staticmethod
    def _heuristic(text: str, lang: str) -> HeadInfo:
        words = text.split()
        if not words:
            return HeadInfo("", "", "HEURISTIC (no tokens)")
        word = words[-1] if lang == "en" else max(words, key=len)
        key = _singularise_en(word.lower()) if lang == "en" else word.lower()
        display = key[:1].upper() + key[1:]
        return HeadInfo(key, display, f"HEURISTIC (Head: {display})")


# ---------------------------------------------------------------------------
# Step 4: promo maths and the basket-expansion gate
# ---------------------------------------------------------------------------

def promo_mechanics(name_a: str, price_a: float, name_c: str, price_c: float,
                    discount_pct: int = DISCOUNT_PCT) -> dict[str, Any]:
    """Higher-priced item is the full-price anchor; the cheaper one is discounted."""
    if price_c > price_a:
        (anchor, anchor_price), (pull, pull_price) = (name_c, price_c), (name_a, price_a)
    else:
        (anchor, anchor_price), (pull, pull_price) = (name_a, price_a), (name_c, price_c)
    discounted = pull_price * (1 - discount_pct / 100)
    return {
        "anchor_item": anchor,
        "discounted_item": pull,
        "discount_pct": discount_pct,
        "regular_bundle_price": round(anchor_price + pull_price, 2),
        "promo_bundle_price": round(anchor_price + discounted, 2),
        # Revenue the bundle adds on top of a shopper who buys only the anchor.
        "net_basket_lift_eur": round(discounted, 2),
        "_anchor_price": anchor_price,
    }


def margin_gate(mech: dict[str, Any], lift: float, min_lift: float) -> tuple[bool, str | None]:
    anchor_price = mech["_anchor_price"]
    floor = anchor_price * MIN_BASKET_EXPANSION
    if mech["promo_bundle_price"] + 1e-9 < round(floor, 2):
        return False, (f"Low basket expansion: bundle €{mech['promo_bundle_price']:.2f} is below "
                       f"110% of the anchor price (€{floor:.2f})")
    if lift < min_lift:
        return False, f"Low lift: {lift:.2f} is below the bundle floor of {min_lift:.2f}"
    return True, None


# ---------------------------------------------------------------------------
# Step 5: flyer copy (Gemma via Ollama, template fallback)
# ---------------------------------------------------------------------------

def display_name(name: str) -> str:
    return smart_title(name) if name.upper() == name else name


def _variant(rule_key: str, n: int) -> int:
    return int(hashlib.md5(rule_key.encode()).hexdigest(), 16) % n


TEMPLATE_TITLES = {
    "de": ["Kombi-Deal der Woche", "Besser zusammen", "Das perfekte Duo", "Zusammen günstiger"],
    "en": ["Combo Deal of the Week", "Better Together", "The Perfect Pair", "Pair Up and Save"],
    "el": ["Προσφορά της εβδομάδας", "Καλύτερα μαζί", "Το τέλειο δίδυμο", "Μαζί πιο οικονομικά"],
}


def template_copy(anchor: str, pull: str, pct: int, rule_key: str) -> dict[str, dict[str, str]]:
    a, b = display_name(anchor), display_name(pull)
    v = _variant(rule_key, len(TEMPLATE_TITLES["de"]))
    return {
        "de": {"title": TEMPLATE_TITLES["de"][v],
               "Copy": f"Kaufe „{a}“ und sichere dir {pct} % Rabatt auf „{b}“."},
        "en": {"title": TEMPLATE_TITLES["en"][v],
               "Copy": f"Buy {a} and get {pct}% off {b}."},
        # «προϊόν» carries the article, so the foreign product name never needs a case ending.
        "el": {"title": TEMPLATE_TITLES["el"][v],
               "Copy": f"Αγοράστε το προϊόν «{a}» και κερδίστε {pct}% έκπτωση στο προϊόν «{b}»."},
    }


def ollama_generate(base_url: str, model: str, prompt: str, *, temperature: float, num_predict: int,
                    timeout: float, seed: int | None = None) -> str:
    """One completion. Raises requests.RequestException / ValueError on failure.

    think=False: gemma4 otherwise spends the token budget on hidden reasoning. No
    "format": "json": in JSON mode gemma4 hit Ollama's token-repeat abort.
    """
    options: dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
    if seed is not None:
        options["seed"] = seed
    r = requests.post(f"{base_url.rstrip('/')}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False, "think": False,
                            "keep_alive": "30m", "options": options},
                      timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "") or ""


def resolve_model(base_url: str, requested: str | None, probe_timeout: float = 3.0) -> str | None:
    """The Ollama model to use, or None (with a log line) when Ollama or Gemma is unavailable."""
    names: list[str] | None = None
    # Two tries: a busy Ollama (still unloading a model) can miss one 3 s probe.
    for attempt in (1, 2):
        try:
            r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=probe_timeout)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            break
        except requests.ConnectionError:
            break  # nothing listening: no point retrying
        except (requests.RequestException, ValueError) as exc:
            log.info("Ollama probe %d failed (%s)", attempt, exc.__class__.__name__)
    if names is None:
        log.warning("Ollama not reachable")
        return None
    wanted = ([requested] if requested else []) + list(PREFERRED_MODELS)
    for cand in wanted:
        for name in (cand, f"{cand}:latest"):
            if name in names:
                return name
    if requested:
        log.warning("Model %s is not pulled in Ollama", requested)
    gemmas = [n for n in names if n.startswith("gemma")]
    if gemmas:
        log.info("none of %s is pulled; using %s", ", ".join(PREFERRED_MODELS), gemmas[0])
        return gemmas[0]
    log.warning("No Gemma model in Ollama")
    return None


LANGUAGE_NAMES = {"de": "German", "en": "English", "el": "Modern Greek"}
LANGUAGE_RULES = {"de": "Write in German and address the reader with du.", "en": "Write in English.",
                  "el": "Write in Modern Greek, in Greek script."}
MAX_COPY_RETRIES = 2
COPY_TEMPERATURE = 0.4
FIRST_PREDICT = 400
REPAIR_PREDICT = 200


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


def parse_all_languages(text: str, langs: tuple[str, ...] = ("de", "en", "el")) -> dict[str, tuple[str, str]]:
    """The readable languages of a multi-language answer; the JSON is cut out of plain text."""
    whole = _loads(text)
    lowered = {str(k).lower(): v for k, v in whole.items()} if isinstance(whole, dict) else {}
    out = {}
    for lang in langs:
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


def _problem_lines(results: dict[str, dict[str, Any]], lang: str, names: tuple[str, ...], pct: int) -> list[str]:
    """One plain instruction per failed check, naming exactly what went wrong."""
    lines = []
    for check in checks.failed_ids(results):
        detail = results[check]["detail"]
        if check == "parse":
            lines.append("Your last answer could not be read. Return only the JSON object.")
        elif check == "names" and results["names"].get("reversed_offer"):
            lines.append(f'You swapped the two products: "{names[0]}" is the full-price item and "{names[1]}" '
                         f"gets the {pct}% discount. Both names must appear verbatim.")
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


def repair_prompt(lang: str, entry: tuple[str, str] | None, results: dict[str, dict[str, Any]],
                  names: tuple[str, ...], pct: int) -> str:
    shown = json.dumps({"anchor_full_price": names[0], "discounted_item": names[1], "discount_pct": pct},
                       ensure_ascii=False)
    previous = f"Previous title: {entry[0]}\nPrevious copy: {entry[1]}\n" if entry else ""
    problems = "\n".join(f"- {line}" for line in _problem_lines(results, lang, names, pct))
    return (
        f"You write supermarket flyer copy in {LANGUAGE_NAMES[lang]}. {LANGUAGE_RULES[lang]} "
        'Fix the text below and return only JSON of the form {"title":"","copy":""}. '
        "title: at most 6 words. copy: one catchy sentence, at most 25 words. "
        f"Do not invent prices or other offers. The copy says the discounted item is {pct}% off when bought "
        f"with the full-price item. Use the product names as given and do not write meta-labels like "
        f"'full-price product' or 'volle Preis-Produkt'.\nBundle: {shown}\n"
        f"{previous}Problems to fix:\n{problems}"
    )


class GemmaCopywriter:
    """Guarded copy (study winner): Gemma, linguistic checks, targeted repair, then template."""

    def __init__(self, base_url: str, model: str | None, enabled: bool,
                 probe_timeout: float = 3.0, gen_timeout: float = 60.0,
                 languages: tuple[str, ...] = ("de", "en")) -> None:
        # Greek stays on templates by default: even guarded Greek text was only 27 %
        # publishable in the study. --gemma-greek adds "el" to languages.
        self.languages = languages
        self.base_url = base_url.rstrip("/")
        self.probe_timeout = probe_timeout
        self.gen_timeout = gen_timeout
        self.stats = {"llm_ok": 0, "template": 0, "calls": 0, "repairs": 0}
        self.model = (resolve_model(self.base_url, model, probe_timeout) if enabled else None)
        if enabled and self.model is None:
            log.warning("Using copy templates")

    def _prompt(self, anchor: str, pull: str, pct: int) -> str:
        bundle = json.dumps({"anchor_full_price": display_name(anchor),
                             "discounted_item": display_name(pull), "discount_pct": pct},
                            ensure_ascii=False)
        langs = self.languages
        shape = ",".join(f'"{l}":{{"title":"","copy":""}}' for l in langs)
        legend = ", ".join({"de": "de = German (address the reader with du)", "en": "en = English",
                            "el": "el = Modern Greek in Greek script"}[l] for l in langs)
        return (
            "You write supermarket flyer copy. Return only JSON of the form "
            f"{{{shape}}}. {legend}. "
            "title: at most 6 words. copy: one catchy sentence, at most 25 words, that says the "
            f"discounted item is {pct}% off when bought with the full-price item. "
            "Use the product names as given and do not write meta-labels like 'full-price product' or "
            "'volle Preis-Produkt'. Do not invent prices or other offers.\n"
            f"Bundle: {bundle}"
        )

    def _call(self, prompt: str, num_predict: int) -> str:
        self.stats["calls"] += 1
        return ollama_generate(self.base_url, self.model or "", prompt, temperature=COPY_TEMPERATURE,
                               num_predict=num_predict, timeout=self.gen_timeout)

    def _guarded(self, lang: str, first: tuple[str, str] | None, names: tuple[str, ...], pct: int,
                 rule_key: str) -> tuple[str, str] | None:
        """Check, repair at most twice; the passing (title, copy) or None (template)."""
        entry = first
        results = checks.run_checks(lang, *(entry or ("", "")), names, pct)
        for retry in range(1, MAX_COPY_RETRIES + 1):
            if not checks.failed_ids(results):
                break
            log.info("Gemma %s copy for %s failed %s: repair %d", lang, rule_key,
                     ",".join(checks.failed_ids(results)), retry)
            try:
                self.stats["repairs"] += 1
                text = self._call(repair_prompt(lang, entry, results, names, pct), REPAIR_PREDICT)
            except (requests.RequestException, ValueError) as exc:
                log.warning("Gemma repair call failed (%s): template used", exc.__class__.__name__)
                return None
            entry = parse_one_language(text, lang)
            results = checks.run_checks(lang, *(entry or ("", "")), names, pct)
        if checks.failed_ids(results) or entry is None:
            return None
        return " ".join(entry[0].split()), " ".join(entry[1].split())

    def write(self, anchor: str, pull: str, pct: int, rule_key: str) -> dict[str, dict[str, str]]:
        copy = template_copy(anchor, pull, pct, rule_key)
        if not self.model:
            self.stats["template"] += 3
            return copy
        try:
            text = self._call(self._prompt(anchor, pull, pct), FIRST_PREDICT)
        except (requests.RequestException, ValueError) as exc:
            log.warning("Gemma call failed for %s (%s): templates used", rule_key, exc.__class__.__name__)
            self.stats["template"] += 3
            return copy
        first = parse_all_languages(text, self.languages)
        names = (display_name(anchor), display_name(pull))   # checks run on the DISPLAY names
        for lang in ("de", "en", "el"):
            if lang not in self.languages:
                self.stats["template"] += 1
                continue
            good = self._guarded(lang, first.get(lang), names, pct, rule_key)
            if good:
                copy[lang] = {"title": good[0], "Copy": good[1]}
                self.stats["llm_ok"] += 1
            else:
                log.info("Gemma %s copy for %s: template used", lang, rule_key)
                self.stats["template"] += 1
        return copy


# ---------------------------------------------------------------------------
# Step 4b: LLM judge (study condition CLf, prompt variant "examples")
# ---------------------------------------------------------------------------
# The prompt text is copied from experiments/triage_prompts.py (variant "examples",
# with the parser analysis); the fixed text comes first and the pair last.

JUDGE_DEFINITIONS = """\
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

JUDGE_EXAMPLES = (
    ("Snorkel Mask", "Swim Fins", "COMPLEMENT"),
    ("Ski Gloves Black", "Ski Gloves Navy", "SUBSTITUTE"),
    ("Car Wax", "Dog Biscuits", "NEITHER"),
)
JUDGE_ONE_WORD = "Answer with exactly one word: COMPLEMENT, SUBSTITUTE or NEITHER."
JUDGE_WORDS = ("COMPLEMENT", "SUBSTITUTE", "NEITHER")
JUDGE_NUM_PREDICT = 12
JUDGE_SEED = 42
JUDGE_MAX_CONSECUTIVE_ERRORS = 3
_JUDGE_WORD_RE = re.compile(r"\b(" + "|".join(JUDGE_WORDS) + r")\b", re.IGNORECASE)


def _judge_words(words: list[str]) -> str:
    return ", ".join(words) if words else "none"


def judge_prompt(name_1: str, name_2: str, cl: dict[str, Any]) -> str:
    examples = "\n".join(f'- "{a}" + "{b}": {v}' for a, b, v in JUDGE_EXAMPLES)
    analysis = (
        "A parser reports:\n"
        f"- Product 1 head noun: {cl['head_a'] or 'unknown'}. Attribute words set aside: {_judge_words(cl['stripped_a'])}.\n"
        f"- Product 2 head noun: {cl['head_b'] or 'unknown'}. Attribute words set aside: {_judge_words(cl['stripped_b'])}.\n"
        f"- Same head noun: {'yes' if cl['same_head'] else 'no'}.\n"
        f"- Shared compound: {cl['compound'] or 'none'}.\n"
        f"- Postage, packaging or other operational term: {cl['stop_term'] or 'none'}.\n"
        "This is automatic evidence only. Decide using the definitions above."
    )
    return "\n\n".join([JUDGE_DEFINITIONS, f"Examples:\n{examples}", JUDGE_ONE_WORD,
                        f"Product 1: {name_1}\nProduct 2: {name_2}", analysis])


def parse_judge_verdict(text: str) -> str:
    """The first of the three words, else INVALID."""
    m = _JUDGE_WORD_RE.search(text or "")
    return m.group(1).upper() if m else "INVALID"


def stripped_words(name: str, learned: frozenset[str], lang: str) -> list[str]:
    """Attribute words normalise_for_nlp removed (not counts, sizes or other number tokens)."""
    bare = lambda tok: tok.lower().strip(".,;:()'\"")  # noqa: E731
    used = learned if lang == "en" else frozenset()
    kept = Counter(bare(t) for t in normalise_for_nlp(name, used).split())
    out = []
    for tok in _COUNT_PHRASE_RE.sub(" ", name).split():
        word = bare(tok)
        if kept[word] > 0:
            kept[word] -= 1
        elif word not in UNIT_TOKENS and any(c.isalpha() for c in tok) and not any(c.isdigit() for c in tok):
            out.append(word)
    return out


def judge_survivors(records: list[dict[str, Any]], base_url: str, model: str, timeout: float
                    ) -> dict[str, int]:
    """Ask the LLM about every record still approved after the rules and the gate.

    Only the verdict COMPLEMENT keeps a record approved; SUBSTITUTE, NEITHER and an
    unreadable answer (INVALID) turn it into STATUS_JUDGE. A failed call says nothing
    about the pair, so that pair keeps the rules' decision and is counted as unjudged.
    After JUDGE_MAX_CONSECUTIVE_ERRORS failures in a row the stage stops (Ollama looks
    down) and the remaining pairs stay approved too, with a warning.
    """
    todo = [r for r in records if r["status"] == STATUS_APPROVED]
    stats = {"calls": 0, "rejected": 0, "errors": 0, "unjudged": 0}
    consecutive = 0
    log.info("LLM judge: %d surviving pairs go to %s", len(todo), model)
    for n, rec in enumerate(todo):
        if consecutive >= JUDGE_MAX_CONSECUTIVE_ERRORS:
            stats["unjudged"] += len(todo) - n
            log.warning("LLM judge stopped after %d failed calls in a row: %d pairs in total stay approved "
                        "WITHOUT an LLM verdict", consecutive, stats["unjudged"])
            break
        c: Candidate = rec["candidate"]
        prompt = judge_prompt(c.antecedent, c.consequent, rec["cl"])
        stats["calls"] += 1
        try:
            verdict = parse_judge_verdict(ollama_generate(base_url, model, prompt, temperature=0.0,
                                                          num_predict=JUDGE_NUM_PREDICT, timeout=timeout,
                                                          seed=JUDGE_SEED))
            consecutive = 0
        except (requests.RequestException, ValueError) as exc:
            log.warning("LLM judge call failed for %s + %s (%s): the pair keeps the rules' decision",
                        c.antecedent, c.consequent, exc.__class__.__name__)
            consecutive += 1
            stats["errors"] += 1
            stats["unjudged"] += 1
            continue
        rec["judge_verdict"] = verdict
        if verdict != "COMPLEMENT":
            rec["status"] = STATUS_JUDGE
            rec["explanation"] = f"LLM judge: {verdict}"
            stats["rejected"] += 1
    return stats


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pruner(cands: list[Candidate], extractor: HeadNounExtractor, min_bundle_lift: float,
               bag_guard: bool = True) -> tuple[list[dict[str, Any]], float]:
    """Apply all three filters to every rule. Returns records and CPU time in ms."""
    t0 = time.perf_counter()
    extractor.prepare([(c.antecedent, c.lang_a) for c in cands] + [(c.consequent, c.lang_c) for c in cands])
    records = []
    for c in cands:
        head_a, head_c = extractor.get(c.antecedent, c.lang_a), extractor.get(c.consequent, c.lang_c)
        stop = stop_item_match(c.antecedent, bag_guard) or stop_item_match(c.consequent, bag_guard)
        german = "de" in (c.lang_a, c.lang_c)
        shared = head_collision(head_a.head_noun, head_c.head_noun, german)
        # The parser's findings as the judge prompt sees them (before the compound
        # head overwrites the items' heads); mirrors experiments/cl_features.py.
        same_head = head_collision(head_a.head_noun, head_c.head_noun, False)
        cl = {"head_a": head_a.display, "head_b": head_c.display, "same_head": same_head,
              "stripped_a": stripped_words(c.antecedent, extractor.learned_variants, c.lang_a),
              "stripped_b": stripped_words(c.consequent, extractor.learned_variants, c.lang_c),
              "stop_term": stop}
        if shared:
            compound = None
        elif german:
            compound = shared_name_words(c.antecedent, c.consequent)
        else:
            compound = open_compound_match(c.antecedent, c.consequent, extractor.learned_variants)
        if german and shared and not same_head:
            cl["compound"] = german_compound(head_a.head_noun, head_c.head_noun)
        else:
            cl["compound"] = compound
        if compound:
            # The shared compound's head (its last word) becomes both items' head.
            head = compound.split()[-1]
            head = _singularise_en(head)
            disp = head[:1].upper() + head[1:]
            head_a = HeadInfo(head, disp, f"{head_a.pos_pattern.split(' (')[0]} (Compound Head: {compound})")
            head_c = HeadInfo(head, disp, f"{head_c.pos_pattern.split(' (')[0]} (Compound Head: {compound})")
            shared = True
        mech = promo_mechanics(c.antecedent, c.price_a, c.consequent, c.price_c)
        passes, margin_reason = margin_gate(mech, c.lift, min_bundle_lift)

        # Every check is computed for every rule (the UI toggles them independently);
        # status is the first failing stage in pipeline order.
        if stop:
            status = STATUS_PACKAGING
            noisy = c.antecedent if stop_item_match(c.antecedent, bag_guard) else c.consequent
            explanation = f'Packaging/operational noise: "{stop}" in "{display_name(noisy)}"'
        elif shared:
            status, explanation = STATUS_HEAD, SUBSTITUTE_EXPLANATION
        elif not passes:
            status, explanation = STATUS_MARGIN, margin_reason
        else:
            status, explanation = STATUS_APPROVED, None

        records.append({
            "candidate": c, "status": status, "explanation": explanation,
            "head_a": head_a, "head_c": head_c, "stop": stop is not None,
            "shared": shared, "passes": passes, "mech": mech, "cl": cl,
        })
    return records, (time.perf_counter() - t0) * 1000


def to_contract(rec: dict[str, Any], rule_id: str, copy: dict | None) -> dict[str, Any]:
    c: Candidate = rec["candidate"]
    mech = {k: v for k, v in rec["mech"].items() if not k.startswith("_")}
    return {
        "id": rule_id,
        "domain": c.domain,
        "antecedent": {"name": c.antecedent, "price_eur": round(c.price_a, 2),
                       "head_noun": rec["head_a"].display, "pos_pattern": rec["head_a"].pos_pattern},
        "consequent": {"name": c.consequent, "price_eur": round(c.price_c, 2),
                       "head_noun": rec["head_c"].display, "pos_pattern": rec["head_c"].pos_pattern},
        "metrics": {"support": round(c.support, 4), "confidence": round(c.confidence, 4),
                    "lift": round(c.lift, 2)},
        "linguistic_checks": {
            "is_packaging_or_stop_item": rec["stop"],
            "shares_head_noun": rec["shared"],
            "passes_margin_gate": rec["passes"],
            "status": rec["status"],
            "rejection_explanation": rec["explanation"],
        },
        "promo_mechanics": mech,
        "flyer_copy": copy,
    }


def build_payload(records: list[dict[str, Any]], prune_ms: float, n_transactions: int,
                  writer: GemmaCopywriter, top_n: int, xray_per_reason: int | None,
                  judge: dict[str, int] | None = None) -> dict[str, Any]:
    """judge = stats of judge_survivors(), or None when the judge stage did not run."""
    by_status: dict[str, list[dict[str, Any]]] = {s: [] for s in
                                                  (STATUS_APPROVED, STATUS_PACKAGING, STATUS_HEAD, STATUS_MARGIN,
                                                   STATUS_JUDGE)}
    for rec in records:
        by_status[rec["status"]].append(rec)
    rank = lambda r: (r["candidate"].lift, r["candidate"].confidence)  # noqa: E731
    for group in by_status.values():
        group.sort(key=rank, reverse=True)

    raw = len(records)
    surviving = len(by_status[STATUS_APPROVED])
    # llm_calls_saved_pct is defined against ALL raw rules: an LLM-only pipeline would make
    # one call per raw rule. We count the calls actually made: judge calls plus every copy
    # call (first call and repairs, all languages). If the judge stage did not run, the
    # pairs that survived the rules count as the calls it would have needed (the
    # rules-only definition, 1 - survivors/raw), plus any copy calls made. The value can
    # be negative if repairs make the guarded copy dearer than one call per rule.
    judged = judge["calls"] if judge is not None else surviving + len(by_status[STATUS_JUDGE])
    telemetry = {
        "transactions_analyzed": int(n_transactions),
        "raw_association_rules": raw,
        "rejected_packaging_noise": len(by_status[STATUS_PACKAGING]),
        "rejected_same_head_noun": len(by_status[STATUS_HEAD]),
        "rejected_low_margin": len(by_status[STATUS_MARGIN]),
        "rejected_llm_judge": len(by_status[STATUS_JUDGE]),
        "surviving_complementary_bundles": surviving,
        "llm_calls_saved_pct": 0.0,   # set below, once the copy calls have been made
        "cpu_pruning_time_ms": round(prune_ms, 1),
    }

    # Copy is generated only for the top-N surviving bundles; a sample of the
    # highest-lift rejects per reason is exported for the X-ray view.
    chosen = by_status[STATUS_APPROVED][:top_n]
    for status in (STATUS_PACKAGING, STATUS_HEAD, STATUS_MARGIN, STATUS_JUDGE):
        group = by_status[status]
        chosen += group if xray_per_reason is None else group[:xray_per_reason]
    chosen.sort(key=rank, reverse=True)

    rules = []
    for i, rec in enumerate(chosen, start=1):
        rule_id = f"rule-{i:02d}"
        copy = None
        if rec["status"] == STATUS_APPROVED:
            m = rec["mech"]
            log.info("Writing copy for %s: %s + %s", rule_id, m["anchor_item"], m["discounted_item"])
            texts = writer.write(m["anchor_item"], m["discounted_item"], m["discount_pct"],
                                 f"{m['anchor_item']}|{m['discounted_item']}")
            copy = {"badge_text": f"-{m['discount_pct']}% KOMBI-DEAL", **texts}
        rules.append(to_contract(rec, rule_id, copy))
    llm_calls = judged + writer.stats.get("calls", 0)
    telemetry["llm_calls_saved_pct"] = round(100 * (1 - llm_calls / raw), 1) if raw else 0.0
    return {"telemetry": telemetry, "rules": rules}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, help="UCI Online Retail .csv or .xlsx (omit to use the built-in sample)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output JSON (default: {DEFAULT_OUT.relative_to(REPO_ROOT)} with --csv, "
                        f"else {SAMPLE_OUT.relative_to(REPO_ROOT)})")
    p.add_argument("--min-support", type=float, default=0.01)
    p.add_argument("--min-lift", type=float, default=1.5, help="association_rules lift threshold")
    p.add_argument("--min-bundle-lift", type=float, default=DEFAULT_MIN_BUNDLE_LIFT,
                   help="lift floor in the margin & lift gate")
    p.add_argument("--gbp-to-eur", type=float, default=1.16, help="UCI prices are GBP (2010–11)")
    p.add_argument("--top-n", type=int, default=12, help="surviving bundles that get flyer copy")
    p.add_argument("--xray-samples", type=int, default=6,
                   help="rejected rules exported per reason in --csv mode (built-in mode exports all)")
    p.add_argument("--strict-spec-regex", action="store_true",
                   help="treat every 'bag' as packaging (removes sold lunch/jumbo bags on UCI)")
    p.add_argument("--gemma-greek", action="store_true",
                   help="also use Gemma for Greek (off by default: see README, Greek quality)")
    p.add_argument("--no-llm", action="store_true", help="skip Ollama: no LLM judge, copy templates")
    p.add_argument("--no-llm-judge", action="store_true",
                   help="skip the LLM judge stage (copy still uses Gemma)")
    p.add_argument("--model", help=f"Ollama model (default: the first pulled of {', '.join(PREFERRED_MODELS)}, "
                                   "else any pulled gemma*)")
    p.add_argument("--ollama-url", default=OLLAMA_URL)
    p.add_argument("--llm-timeout", type=float, default=60.0,
                   help="seconds per generation call (the 3 s timeout is used for the availability probe)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = DEFAULT_OUT if args.csv else SAMPLE_OUT
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(message)s", stream=sys.stderr)
    t_start = time.perf_counter()

    if args.csv:
        if not args.csv.exists():
            log.error("File not found: %s", args.csv)
            return 2
        cands, n_tx, catalog = load_uci_candidates(args.csv, args.min_support, args.min_lift,
                                                   args.gbp_to_eur)
        xray = args.xray_samples
    else:
        log.info("No --csv given: using the built-in %d-rule sample", len(BUILTIN_RULES))
        cands, n_tx, xray = list(BUILTIN_RULES), 0, None
        catalog = []

    # Catalog-wide minimal pairs (CSV mode); learning is not counted as pruning time
    # because it runs once per catalog, not per rule.
    learned = learn_variant_attributes(catalog) if catalog else frozenset()
    if learned:
        log.info("Learned %d variant attributes from %d catalog names", len(learned), len(catalog))
    extractor = HeadNounExtractor(learned)
    extractor.load({c.lang_a for c in cands} | {c.lang_c for c in cands})
    records, prune_ms = run_pruner(cands, extractor, args.min_bundle_lift,
                                   bag_guard=not args.strict_spec_regex)

    writer = GemmaCopywriter(args.ollama_url, args.model, enabled=not args.no_llm,
                             gen_timeout=args.llm_timeout,
                             languages=("de", "en", "el") if args.gemma_greek else ("de", "en"))
    judge_stats = None
    if args.no_llm:
        log.info("LLM judge skipped: --no-llm")
    elif args.no_llm_judge:
        log.info("LLM judge skipped: --no-llm-judge")
    elif not writer.model:
        log.info("LLM judge skipped: Ollama or Gemma is not available (rules-only result)")
    else:
        judge_stats = judge_survivors(records, writer.base_url, writer.model, args.llm_timeout)
    payload = build_payload(records, prune_ms, n_tx, writer, args.top_n, xray, judge_stats)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    t = payload["telemetry"]
    print(f"\nSmart Flyer ({'UCI ' + args.csv.name if args.csv else 'built-in sample'})")
    print(f"  transactions analysed      {t['transactions_analyzed']:>8,}")
    print(f"  raw 1->1 rules             {t['raw_association_rules']:>8,}")
    print(f"  - packaging / stop items   {t['rejected_packaging_noise']:>8,}")
    print(f"  - shared head noun         {t['rejected_same_head_noun']:>8,}")
    print(f"  - margin & lift gate       {t['rejected_low_margin']:>8,}")
    print(f"  - LLM judge                {t['rejected_llm_judge']:>8,}")
    print(f"  = surviving bundles        {t['surviving_complementary_bundles']:>8,}")
    print(f"  LLM calls saved            {t['llm_calls_saved_pct']:>7.1f}%")
    print(f"  CPU pruning time           {t['cpu_pruning_time_ms']:>7.1f} ms")
    if judge_stats:
        print(f"  judge: {judge_stats['calls']} calls, {judge_stats['rejected']} rejected, "
              f"{judge_stats['errors']} errors, {judge_stats['unjudged']} unjudged")
    print(f"  copy calls: {writer.stats['calls']} ({writer.stats['repairs']} repairs)")
    print(f"  copy: model={writer.model or 'none'}  "
          f"llm={writer.stats['llm_ok']} template={writer.stats['template']} (per language)")
    print(f"  wrote {len(payload['rules'])} rules to {args.out}  ({time.perf_counter() - t_start:.1f} s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
