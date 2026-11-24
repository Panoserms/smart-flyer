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
  5. Copy: local Gemma through Ollama writes DE / EN / EL flyer copy for the top
     surviving bundles. Each language falls back to a fixed template when Ollama is
     not running or the model's answer fails validation.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

log = logging.getLogger("smart_flyer")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "public" / "smart_flyer_data.json"

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
    "dolly", "skulls", "strawberry", "union", "jack", "rose", "roses",
    # marketing labels (DE grocery)
    "bio", "barista", "classic", "original", "premium", "extra", "ungesüßt",
}
# German catalog nouns given with an inflected base: Cherrytomaten -> Cherrytomate.
GERMAN_FRAME = "Ich kaufe"
GERMAN_LEMMA_BASES = ("tomate", "bohne", "banane", "nudel", "kartoffel", "zwiebel",
                      "gurke", "orange", "zitrone", "erdbeere", "möhre", "karotte")

# Compound heads: the spec rule is "last 5+ characters match". German grocery
# compounds also end in common 2–4 letter bases (Apfelsaft / Orangensaft) that the
# 5-character rule misses, so a small lexicon of short bases is checked as well.
MIN_COMPOUND_SUFFIX = 5
SHORT_COMPOUND_BASES = ("saft", "wein", "bier", "brot", "käse", "reis", "mehl", "tee", "öl")
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
PREFERRED_MODELS = ("gemma2:2b", "gemma:2b")
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
# Letters outside Latin / Greek (gemma4 once emitted Thai inside Greek copy).
FOREIGN_SCRIPT_RE = re.compile(r"[^\W\d_A-Za-z\u00C0-\u024F\u0370-\u03FF\u1F00-\u1FFF]")

STATUS_APPROVED = "approved_bundle"
STATUS_PACKAGING = "rejected_packaging_noise"
STATUS_HEAD = "rejected_same_head_noun"
STATUS_MARGIN = "rejected_low_margin"


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
    DIET METAL SIGN. Two kinds of shared open compound mark a variant pair:
      (a) the same leading or trailing two words, unless both are attribute words
          (RED RETROSPOT ... / RED RETROSPOT ...);
      (b) the same three words anywhere (CHOCOLATE HOT WATER BOTTLE / HOT WATER
          BOTTLE KEEP CALM), unless any of them is an attribute (RED SPOTTY PAPER
          PLATES / RED SPOTTY PAPER CUPS are complements).
    """
    attrs = VARIANT_ATTRIBUTES | learned_variants
    a, b = _catalog_tokens(name_a), _catalog_tokens(name_b)
    if len(a) < 2 or len(b) < 2 or a == b:
        return None
    for pa, pb in ((a[:2], b[:2]), (a[-2:], b[-2:])):
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


class GemmaCopywriter:
    def __init__(self, base_url: str, model: str | None, enabled: bool,
                 probe_timeout: float = 3.0, gen_timeout: float = 60.0,
                 languages: tuple[str, ...] = ("de", "en")) -> None:
        # Greek is off by default: on the first gemma4:e2b run, 6 of 18 Greek texts had
        # invented words, a translated or wrong product, or Thai script.
        self.languages = languages
        self.base_url = base_url.rstrip("/")
        self.probe_timeout = probe_timeout
        self.gen_timeout = gen_timeout
        self.stats = {"llm_ok": 0, "template": 0}
        self.model = self._resolve(model) if enabled else None

    def _resolve(self, requested: str | None) -> str | None:
        names: list[str] | None = None
        # Two tries: a busy Ollama (still unloading a model) can miss one 3 s probe.
        for attempt in (1, 2):
            try:
                r = requests.get(f"{self.base_url}/api/tags", timeout=self.probe_timeout)
                r.raise_for_status()
                names = [m.get("name", "") for m in r.json().get("models", [])]
                break
            except requests.ConnectionError:
                break  # nothing listening: no point retrying
            except (requests.RequestException, ValueError) as exc:
                log.info("Ollama probe %d failed (%s)", attempt, exc.__class__.__name__)
        if names is None:
            log.warning("Ollama not reachable: using copy templates")
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
            log.info("gemma2:2b / gemma:2b not pulled; using %s", gemmas[0])
            return gemmas[0]
        log.warning("No Gemma model in Ollama: using copy templates")
        return None

    def _prompt(self, anchor: str, pull: str, pct: int) -> str:
        bundle = json.dumps({"anchor_full_price": display_name(anchor),
                             "discounted_item": display_name(pull), "discount_pct": pct},
                            ensure_ascii=False)
        return (
            "You write supermarket flyer copy. Return only JSON of the form "
            '{"de":{"title":"","copy":""},"en":{"title":"","copy":""},"el":{"title":"","copy":""}}. '
            "de = German (address the reader with du), en = English, el = Modern Greek in Greek script. "
            "title: at most 6 words. copy: one catchy sentence, at most 25 words, that says the "
            f"discounted item is {pct}% off when bought with the full-price item. "
            "Use the product names as given. Do not invent prices or other offers.\n"
            f"Bundle: {bundle}"
        )

    @staticmethod
    def _valid(lang: str, entry: Any, pct: int, names: tuple[str, ...] = ()) -> bool:
        """Cheap faithfulness checks; they cannot judge grammar or tone."""
        if not isinstance(entry, dict):
            return False
        title, copy = entry.get("title"), entry.get("copy") or entry.get("Copy")
        if not (isinstance(title, str) and isinstance(copy, str)):
            return False
        title, copy = title.strip(), copy.strip()
        if not (0 < len(title) <= 70 and 10 <= len(copy) <= 260 and str(pct) in copy):
            return False
        if FOREIGN_SCRIPT_RE.search(title + copy):
            return False
        # Both products must be named exactly as on the shelf label (no translated,
        # shortened or invented product names).
        flat = " ".join(copy.split()).casefold()
        if any(" ".join(n.split()).casefold() not in flat for n in names):
            return False
        greek = len(GREEK_RE.findall(title + copy))
        return greek >= 10 if lang == "el" else greek == 0

    def write(self, anchor: str, pull: str, pct: int, rule_key: str) -> dict[str, dict[str, str]]:
        copy = template_copy(anchor, pull, pct, rule_key)
        if not self.model:
            self.stats["template"] += 3
            return copy
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": self._prompt(anchor, pull, pct),
                      # think=False: gemma4 otherwise spends the token budget on hidden
                      # reasoning. No "format": "json": in JSON mode gemma4 hit Ollama's
                      # token-repeat abort, so the JSON object is cut out of plain text.
                      "stream": False, "think": False,
                      "options": {"temperature": 0.4, "num_predict": 400}},
                timeout=self.gen_timeout,
            )
            r.raise_for_status()
            text = r.json().get("response", "") or ""
            start, end = text.find("{"), text.rfind("}")
            answer = json.loads(text[start:end + 1]) if 0 <= start < end else {}
        except (requests.RequestException, ValueError) as exc:
            log.warning("Gemma call failed for %s (%s): templates used", rule_key, exc.__class__.__name__)
            self.stats["template"] += 3
            return copy
        names = (display_name(anchor), display_name(pull))
        for lang in ("de", "en", "el"):
            if lang not in self.languages:
                self.stats["template"] += 1
                continue
            entry = answer.get(lang) if isinstance(answer, dict) else None
            if self._valid(lang, entry, pct, names):
                copy[lang] = {"title": " ".join(entry["title"].split()),
                              "Copy": " ".join((entry.get("copy") or entry.get("Copy")).split())}
                self.stats["llm_ok"] += 1
            else:
                log.info("Gemma %s copy for %s failed validation: template used", lang, rule_key)
                self.stats["template"] += 1
        return copy


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
        if shared:
            compound = None
        elif german:
            compound = shared_name_words(c.antecedent, c.consequent)
        else:
            compound = open_compound_match(c.antecedent, c.consequent, extractor.learned_variants)
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
            "shared": shared, "passes": passes, "mech": mech,
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
                  writer: GemmaCopywriter, top_n: int, xray_per_reason: int | None) -> dict[str, Any]:
    by_status: dict[str, list[dict[str, Any]]] = {s: [] for s in
                                                  (STATUS_APPROVED, STATUS_PACKAGING, STATUS_HEAD, STATUS_MARGIN)}
    for rec in records:
        by_status[rec["status"]].append(rec)
    rank = lambda r: (r["candidate"].lift, r["candidate"].confidence)  # noqa: E731
    for group in by_status.values():
        group.sort(key=rank, reverse=True)

    raw = len(records)
    surviving = len(by_status[STATUS_APPROVED])
    telemetry = {
        "transactions_analyzed": int(n_transactions),
        "raw_association_rules": raw,
        "rejected_packaging_noise": len(by_status[STATUS_PACKAGING]),
        "rejected_same_head_noun": len(by_status[STATUS_HEAD]),
        "rejected_low_margin": len(by_status[STATUS_MARGIN]),
        "surviving_complementary_bundles": surviving,
        "llm_calls_saved_pct": round(100 * (1 - surviving / raw), 1) if raw else 0.0,
        "cpu_pruning_time_ms": round(prune_ms, 1),
    }

    # Copy is generated only for the top-N surviving bundles; a sample of the
    # highest-lift rejects per reason is exported for the X-ray view.
    chosen = by_status[STATUS_APPROVED][:top_n]
    for status in (STATUS_PACKAGING, STATUS_HEAD, STATUS_MARGIN):
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
    return {"telemetry": telemetry, "rules": rules}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, help="UCI Online Retail .csv or .xlsx (omit to use the built-in sample)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output JSON (default: {DEFAULT_OUT})")
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
    p.add_argument("--no-llm", action="store_true", help="skip Ollama and use copy templates")
    p.add_argument("--model", help="Ollama model (default: gemma2:2b, gemma:2b, else any pulled gemma*)")
    p.add_argument("--ollama-url", default=OLLAMA_URL)
    p.add_argument("--llm-timeout", type=float, default=60.0,
                   help="seconds per generation call (the 3 s timeout is used for the availability probe)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


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
    payload = build_payload(records, prune_ms, n_tx, writer, args.top_n, xray)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    t = payload["telemetry"]
    print(f"\nSmart Flyer ({'UCI ' + args.csv.name if args.csv else 'built-in sample'})")
    print(f"  transactions analysed      {t['transactions_analyzed']:>8,}")
    print(f"  raw 1->1 rules             {t['raw_association_rules']:>8,}")
    print(f"  - packaging / stop items   {t['rejected_packaging_noise']:>8,}")
    print(f"  - shared head noun         {t['rejected_same_head_noun']:>8,}")
    print(f"  - margin & lift gate       {t['rejected_low_margin']:>8,}")
    print(f"  = surviving bundles        {t['surviving_complementary_bundles']:>8,}")
    print(f"  LLM calls saved            {t['llm_calls_saved_pct']:>7.1f}%")
    print(f"  CPU pruning time           {t['cpu_pruning_time_ms']:>7.1f} ms")
    print(f"  copy: model={writer.model or 'none'}  "
          f"llm={writer.stats['llm_ok']} template={writer.stats['template']} (per language)")
    print(f"  wrote {len(payload['rules'])} rules to {args.out}  ({time.perf_counter() - t_start:.1f} s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
