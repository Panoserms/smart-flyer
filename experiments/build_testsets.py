#!/usr/bin/env python3
"""Build the Smart Flyer study's data sets: seeded, re-runnable, blind by construction.

Run from the repo root, never from inside a download directory:

    .venv/bin/python -I experiments/build_testsets.py            # dev, t1, t2
    .venv/bin/python -I experiments/build_testsets.py --german   # dev_de, t3

Writes data/testsets/pairs_<set>.csv, catalog_<set>.txt (dev, t1, t2), manifest.json and
eval/gold/blind/{A,B}_<set>.csv. Downloads stay in their own directories under data/raw/
and are only ever read as data. Output is counts, ids and hashes, never a test row.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import random
import re
import shutil
import statistics
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

# pipeline/ is read-only for this script, and -I ignores PYTHONDONTWRITEBYTECODE.
sys.dont_write_bytecode = True

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
# -I keeps the script and current directories off sys.path. Appended so that nothing in
# pipeline/ can shadow the standard library.
sys.path.append(str(REPO_ROOT / "pipeline"))

from export_smart_flyer import Candidate, _read_table, clean_uci, load_uci_candidates, mine_rules  # noqa: E402

log = logging.getLogger("build_testsets")

SEED = 20261106
MIN_SUPPORT = 0.01
MIN_LIFT = 1.5
GBP_TO_EUR = 1.16
CAP = 250

DEV_EXPECTED_PAIRS = 300
DEV_EXPECTED_BASKETS = 18_402  # README, "Results on UCI Online Retail"
DEV_SOURCE_URL = "https://archive.ics.uci.edu/dataset/352/online+retail"

SHEET = "Year 2009-2010"
SHEET_CSV = "year_2009_2010.csv"
SHEET_COLUMNS = ("Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country")
SHEET_EXPECTED_ROWS = 525_461  # size of this sheet on UCI

GROCERIES_BASKETS = 9_835
GROCERIES_ITEMS = 169
GROCERIES_ITEM_TOTAL = 43_367  # arules: 9835 x 169 x density 0.02609146

SIDECAR = "SOURCE.json"
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
MAX_XLSX_BYTES = 200 * 1024 * 1024

PAIR_COLUMNS = ["pair_id", "set", "a", "b", "support", "confidence", "lift", "price_a", "price_b", "lang_a", "lang_b"]
BLIND_COLUMNS = ["pair_id", "product_1", "product_2"]
ANNOTATORS = ("A", "B")
SET_ORDER = ("dev", "dev_de", "t1", "t2", "t3")
SETS_WITH_CATALOG = ("dev", "t1", "t2")

# Set -> (file in the German directory, row count from the pre-registration).
GERMAN_INPUTS = {"dev_de": ("pairs_dev_de.csv", 30), "t3": ("pairs_t3.csv", 120)}
GERMAN_COLUMNS = {"pair_id", "a", "b", "price_a", "price_b", "design_label"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pair:
    """One unordered product pair in the common format. Missing values are None."""
    a: str
    b: str
    support: float | None = None
    confidence: float | None = None
    lift: float | None = None
    price_a: float | None = None
    price_b: float | None = None


def rng(*parts: str) -> random.Random:
    """A generator for one purpose. String seeds do not depend on the hash seed."""
    return random.Random("/".join((str(SEED), *parts)))


def sort_key(pair: Pair) -> tuple[str, str]:
    return (min(pair.a, pair.b), max(pair.a, pair.b))


def overlap_key(a: str, b: str) -> frozenset[str]:
    """Identity of a pair across sheets: unordered, ignoring letter case."""
    return frozenset((a.casefold(), b.casefold()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, header: list[str], rows: Iterable[list[str]]) -> None:
    """Write, then rename: other runs may read these files at any moment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


# ---------------------------------------------------------------------------
# Downloads: untrusted data, each in its own directory
# ---------------------------------------------------------------------------

def single_xlsx_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    members = [m for m in archive.infolist()
               if m.filename.lower().endswith(".xlsx") and "__MACOSX" not in m.filename]
    if len(members) != 1:
        raise ValueError(f"expected one .xlsx in the zip, found {len(members)}")
    if members[0].file_size > MAX_XLSX_BYTES:
        raise ValueError(f"the .xlsx inflates to {members[0].file_size} bytes, over the limit")
    return members[0]


def validate_retail_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        single_xlsx_member(archive)


def read_groceries(path: Path) -> list[list[str]]:
    """One basket per line. Names stay as they are in the file."""
    with path.open(encoding="utf-8", newline="") as handle:
        baskets = [[name for name in line if name] for line in csv.reader(handle)]
    return [basket for basket in baskets if basket]


def validate_groceries(path: Path) -> None:
    baskets = read_groceries(path)
    items = {name for basket in baskets for name in basket}
    total = sum(len(basket) for basket in baskets)
    if (len(baskets), len(items), total) != (GROCERIES_BASKETS, GROCERIES_ITEMS, GROCERIES_ITEM_TOTAL):
        raise ValueError(f"{len(baskets)} baskets, {len(items)} distinct items, {total} item lines; expected "
                         f"{GROCERIES_BASKETS}, {GROCERIES_ITEMS}, {GROCERIES_ITEM_TOTAL}")


@dataclass(frozen=True)
class Source:
    url: str
    directory: str
    filename: str
    sha256: str  # the exact bytes the published sets were built from
    validate: Callable[[Path], None]


SOURCES = {
    "t1": Source(
        url="https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        directory="online_retail_ii", filename="online_retail_ii.zip",
        sha256="572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb", validate=validate_retail_zip),
    "t2": Source(
        url="https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv",
        directory="groceries", filename="groceries.csv",
        sha256="ff1be892fd6b9b57d1a7bc50de067798963dda607619645988b21789bf23ae3b", validate=validate_groceries),
}


def download(url: str, target: Path) -> None:
    part = target.with_name(target.name + ".part")
    size = 0
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with part.open("wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"download is over {MAX_DOWNLOAD_BYTES} bytes")
                    handle.write(chunk)
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)


def write_sidecar(directory: Path, info: dict[str, Any]) -> None:
    write_text(directory / SIDECAR, json.dumps(info, indent=2, ensure_ascii=False) + "\n")


def fetch(source: Source, raw_dir: Path) -> tuple[Path, dict[str, Any]]:
    """The local copy and its provenance. Downloads once, verifies on every run."""
    directory = raw_dir / source.directory
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.filename
    sidecar = directory / SIDECAR
    if target.exists() and sidecar.exists():
        digest = sha256_file(target)
        if digest != source.sha256:
            raise ValueError(f"{repo_relative(target)} has changed: SHA-256 {digest}, pinned {source.sha256}")
        log.info("Using %s (SHA-256 verified)", repo_relative(target))
        return target, json.loads(sidecar.read_text(encoding="utf-8"))

    log.info("Downloading %s", source.url)
    try:
        download(source.url, target)
        digest = sha256_file(target)
        if digest != source.sha256:
            raise ValueError(f"{source.url}: SHA-256 {digest} differs from the pinned {source.sha256}")
        source.validate(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    info = {"url": source.url, "sha256": digest, "retrieved": date.today().isoformat()}
    write_sidecar(directory, info)
    return target, info


def _cell(value: Any, is_code: bool) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if is_code and isinstance(value, float) and value.is_integer():
        return str(int(value))  # invoice and stock codes are identifiers, not numbers
    return str(value)


def sheet_to_csv(zip_path: Path, csv_path: Path) -> dict[str, Any]:
    """Stream the one sheet to CSV. The other sheet is never parsed; memory stays flat."""
    import openpyxl

    xlsx_path = csv_path.with_name("_extracted.xlsx")
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    date_col = SHEET_COLUMNS.index("InvoiceDate")
    code_cols = (SHEET_COLUMNS.index("Invoice"), SHEET_COLUMNS.index("StockCode"))
    n_rows = 0
    first: datetime | None = None
    last: datetime | None = None
    try:
        with zipfile.ZipFile(zip_path) as archive, xlsx_path.open("wb") as out:
            with archive.open(single_xlsx_member(archive)) as member:
                shutil.copyfileobj(member, out, 1 << 20)
        book = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        try:
            if SHEET not in book.sheetnames:
                raise ValueError(f"sheet {SHEET!r} is missing; found {book.sheetnames}")
            rows = book[SHEET].iter_rows(values_only=True)
            header = tuple(next(rows))[:len(SHEET_COLUMNS)]
            if header != SHEET_COLUMNS:
                raise ValueError(f"unexpected header {header}")
            with tmp.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(SHEET_COLUMNS)
                for row in rows:
                    row = row[:len(SHEET_COLUMNS)]
                    if all(cell is None for cell in row):
                        continue
                    stamp = row[date_col]
                    if not isinstance(stamp, datetime):
                        raise ValueError(f"data row {n_rows + 1}: InvoiceDate is not a date")
                    first = stamp if first is None or stamp < first else first
                    last = stamp if last is None or stamp > last else last
                    writer.writerow([_cell(cell, i in code_cols) for i, cell in enumerate(row)])
                    n_rows += 1
        finally:
            book.close()
        if n_rows != SHEET_EXPECTED_ROWS:
            raise ValueError(f"sheet has {n_rows} data rows, expected {SHEET_EXPECTED_ROWS}")
        tmp.replace(csv_path)
    finally:
        xlsx_path.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return {"name": SHEET, "rows": n_rows, "first_invoice_date": first.date().isoformat(),
            "last_invoice_date": last.date().isoformat(), "csv_sha256": sha256_file(csv_path)}


def cached_sheet(zip_path: Path, info: dict[str, Any]) -> Path:
    csv_path = zip_path.with_name(SHEET_CSV)
    sheet = info.get("sheet")
    if csv_path.exists() and sheet and sha256_file(csv_path) == sheet["csv_sha256"]:
        log.info("Using the cached sheet %s", repo_relative(csv_path))
        return csv_path
    log.info("Streaming sheet %r to CSV", SHEET)
    info["sheet"] = sheet_to_csv(zip_path, csv_path)
    write_sidecar(zip_path.parent, info)
    return csv_path


# ---------------------------------------------------------------------------
# Mining results to pairs
# ---------------------------------------------------------------------------

def settle_orientation(antecedent: str, consequent: str, support: float, confidence: float, lift: float,
                       price_a: float | None, price_c: float | None) -> Pair:
    """Build a Pair, fixing which product is `a` when both directions tie on confidence.

    On a tie the pipeline keeps the direction it meets first. That order follows frozenset
    iteration, so it changes with the per-process string hash, and -I ignores PYTHONHASHSEED.
    Name order settles the tie. supp(a) = support / confidence, supp(b) = confidence / lift.
    """
    tied = math.isclose(support / confidence, confidence / lift, rel_tol=1e-9)
    if tied and antecedent > consequent:
        antecedent, consequent, price_a, price_c = consequent, antecedent, price_c, price_a
    return Pair(antecedent, consequent, support, confidence, lift, price_a, price_c)


def pair_from_candidate(cand: Candidate) -> Pair:
    return settle_orientation(cand.antecedent, cand.consequent, cand.support, cand.confidence, cand.lift,
                              cand.price_a, cand.price_c)


def pairs_from_rules(rules: pd.DataFrame) -> list[Pair]:
    """Same 1->1 filter and unordered dedupe as load_uci_candidates, for data without prices."""
    one_to_one = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)]
    best: dict[frozenset[str], tuple[str, str, float, float, float]] = {}
    for row in one_to_one.itertuples(index=False):
        a, c = next(iter(row.antecedents)), next(iter(row.consequents))
        key = frozenset((a, c))
        if key not in best or float(row.confidence) > best[key][3]:
            best[key] = (a, c, float(row.support), float(row.confidence), float(row.lift))
    return [settle_orientation(a, c, support, confidence, lift, None, None)
            for a, c, support, confidence, lift in best.values()]


# ---------------------------------------------------------------------------
# Writing one set
# ---------------------------------------------------------------------------

def _number(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _price(value: float | None) -> str:
    return "" if value is None else f"{float(value):.2f}"


def order_for_publication(set_name: str, pairs: list[Pair], cap: int | None) -> list[Pair]:
    """Seeded cap, then seeded shuffle, so that a pair id says nothing about lift or relation.

    The canonical sort first makes the result depend only on which pairs exist, not on the
    order the miner (hash-dependent) or a data file happened to give them in.
    """
    chosen = sorted(pairs, key=sort_key)
    if cap is not None and len(chosen) > cap:
        chosen = rng(set_name, "sample").sample(chosen, cap)
    rng(set_name, "ids").shuffle(chosen)
    return chosen


def write_pairs(set_name: str, ordered: list[Pair], lang: str, testsets_dir: Path, blind_dir: Path) -> list[str]:
    """Write pairs_<set>.csv and both blind lists. Returns the pair ids."""
    ids = [f"{set_name}-{i:04d}" for i in range(1, len(ordered) + 1)]
    rows = [[pair_id, set_name, p.a, p.b, _number(p.support), _number(p.confidence), _number(p.lift),
             _price(p.price_a), _price(p.price_b), lang, lang]
            for pair_id, p in zip(ids, ordered)]
    write_csv(testsets_dir / f"pairs_{set_name}.csv", PAIR_COLUMNS, rows)

    for annotator in ANNOTATORS:
        indices = list(range(len(ordered)))
        rng(set_name, annotator, "rows").shuffle(indices)
        sides = rng(set_name, annotator, "sides")
        blind_rows = []
        for i in indices:
            p = ordered[i]
            first, second = (p.a, p.b) if sides.random() < 0.5 else (p.b, p.a)
            blind_rows.append([ids[i], first, second])
        write_csv(blind_dir / f"{annotator}_{set_name}.csv", BLIND_COLUMNS, blind_rows)
    return ids


def write_catalog(set_name: str, names: Iterable[str], testsets_dir: Path) -> int:
    distinct = sorted(set(names))
    write_text(testsets_dir / f"catalog_{set_name}.txt", "".join(f"{name}\n" for name in distinct))
    return len(distinct)


# ---------------------------------------------------------------------------
# dev, t1, t2
# ---------------------------------------------------------------------------

def mine_uci(csv_path: Path) -> tuple[list[Pair], int, list[str]]:
    candidates, n_baskets, catalog = load_uci_candidates(csv_path, MIN_SUPPORT, MIN_LIFT, GBP_TO_EUR)
    return [pair_from_candidate(c) for c in candidates], n_baskets, catalog


def build_dev(args: argparse.Namespace) -> tuple[list[Pair], dict[str, Any]]:
    csv_path = args.raw_dir / "OnlineRetail.csv"
    pairs, n_baskets, catalog = mine_uci(csv_path)
    if len(pairs) != DEV_EXPECTED_PAIRS:
        raise SystemExit(f"STOP: dev gives {len(pairs)} distinct 1->1 pairs, expected {DEV_EXPECTED_PAIRS}. "
                         "Nothing was written.")
    if n_baskets != DEV_EXPECTED_BASKETS:
        log.warning("dev has %d baskets after cleaning; the README says %d", n_baskets, DEV_EXPECTED_BASKETS)
    ordered = order_for_publication("dev", pairs, cap=None)
    write_pairs("dev", ordered, "en", args.testsets_dir, args.blind_dir)
    catalog_n = write_catalog("dev", catalog, args.testsets_dir)
    return pairs, {
        "source_url": DEV_SOURCE_URL, "source_file": repo_relative(csv_path), "file_sha256": sha256_file(csv_path),
        "baskets_after_cleaning": n_baskets, "pairs_1to1_before_dedupe": len(pairs), "overlap_removed": 0,
        "final_n": len(ordered), "seed": SEED, "catalog_n": catalog_n,
    }


def baskets_and_stock_codes(csv_path: Path) -> tuple[set[str], dict[str, set[str]]]:
    df = clean_uci(_read_table(csv_path))
    codes: dict[str, set[str]] = {}
    for name, code in df[["Description", "StockCode"]].drop_duplicates().itertuples(index=False):
        codes.setdefault(name, set()).add(str(code).strip().upper())
    return set(df["InvoiceNo"].unique().tolist()), codes


def audit_t1_against_dev(dev_csv: Path, sheet_csv: Path, dev_pairs: list[Pair], ordered: list[Pair], ids: list[str],
                         testsets_dir: Path) -> dict[str, int]:
    """Measure what the name-based overlap rule cannot see. Changes no set.

    The two UCI sheets overlap in time, and a product renamed between them keeps its stock
    code, so a t1 pair can pass the rule while being the same two products as a dev pair.
    Flagged ids (no names) go to t1_dev_audit.csv, so results can be reported without them.
    """
    dev_baskets, dev_codes = baskets_and_stock_codes(dev_csv)
    gc.collect()
    t1_baskets, t1_codes = baskets_and_stock_codes(sheet_csv)
    gc.collect()

    def squash(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.casefold())

    dev_text = {frozenset((squash(p.a), squash(p.b))) for p in dev_pairs}
    dev_code_pairs = [(dev_codes[p.a], dev_codes[p.b]) for p in dev_pairs]
    flagged = []
    for pair_id, p in zip(ids, ordered):
        codes_a, codes_b = t1_codes[p.a], t1_codes[p.b]
        same_codes = any((codes_a & x and codes_b & y) or (codes_a & y and codes_b & x) for x, y in dev_code_pairs)
        same_text = frozenset((squash(p.a), squash(p.b))) in dev_text
        if same_codes or same_text:
            flagged.append([pair_id, str(int(same_codes)), str(int(same_text))])
    write_csv(testsets_dir / "t1_dev_audit.csv", ["pair_id", "same_stock_codes", "same_text_ignoring_punctuation"],
              sorted(flagged))
    return {
        "baskets_shared_with_dev": len(dev_baskets & t1_baskets),
        "pairs_same_stock_codes_as_dev": sum(row[1] == "1" for row in flagged),
        "pairs_same_text_as_dev_ignoring_punctuation": sum(row[2] == "1" for row in flagged),
    }


def build_t1(args: argparse.Namespace, dev_pairs: list[Pair]) -> dict[str, Any]:
    zip_path, info = fetch(SOURCES["t1"], args.raw_dir)
    sheet_csv = cached_sheet(zip_path, info)
    pairs, n_baskets, catalog = mine_uci(sheet_csv)

    dev_exact = {frozenset((p.a, p.b)) for p in dev_pairs}
    dev_keys = {overlap_key(p.a, p.b) for p in dev_pairs}
    new_pairs = [p for p in pairs if overlap_key(p.a, p.b) not in dev_keys]
    overlap_exact = sum(frozenset((p.a, p.b)) in dev_exact for p in pairs)

    ordered = order_for_publication("t1", new_pairs, cap=CAP)
    ids = write_pairs("t1", ordered, "en", args.testsets_dir, args.blind_dir)
    catalog_n = write_catalog("t1", catalog, args.testsets_dir)
    return {
        "source_url": info["url"], "source_file": repo_relative(zip_path), "file_sha256": info["sha256"],
        "retrieved": info["retrieved"], "sheet": info["sheet"],
        "baskets_after_cleaning": n_baskets, "pairs_1to1_before_dedupe": len(pairs),
        "overlap_removed": len(pairs) - len(new_pairs), "overlap_removed_exact_case": overlap_exact,
        "pairs_after_overlap": len(new_pairs), "final_n": len(ordered), "seed": SEED, "catalog_n": catalog_n,
        **audit_t1_against_dev(args.raw_dir / "OnlineRetail.csv", sheet_csv, dev_pairs, ordered, ids,
                               args.testsets_dir),
    }


def build_t2(args: argparse.Namespace) -> dict[str, Any]:
    path, info = fetch(SOURCES["t2"], args.raw_dir)
    baskets = read_groceries(path)
    df = pd.DataFrame([(str(i), name) for i, basket in enumerate(baskets) for name in basket],
                      columns=["InvoiceNo", "Description"])
    rules = mine_rules(df, MIN_SUPPORT, MIN_LIFT)
    if rules.empty:
        raise ValueError("mining returned no rules")
    pairs = pairs_from_rules(rules)
    ordered = order_for_publication("t2", pairs, cap=CAP)
    write_pairs("t2", ordered, "en", args.testsets_dir, args.blind_dir)
    catalog_n = write_catalog("t2", df["Description"].tolist(), args.testsets_dir)
    return {
        "source_url": info["url"], "source_file": repo_relative(path), "file_sha256": info["sha256"],
        "retrieved": info["retrieved"],
        "baskets_after_cleaning": len(baskets), "pairs_1to1_before_dedupe": len(pairs), "overlap_removed": 0,
        "final_n": len(ordered), "seed": SEED, "catalog_n": catalog_n,
    }


def build_main(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    failures = []
    dev_pairs, manifest["sets"]["dev"] = build_dev(args)
    write_manifest(manifest, args.testsets_dir)
    gc.collect()
    for set_name, builder in (("t1", lambda: build_t1(args, dev_pairs)), ("t2", lambda: build_t2(args))):
        try:
            manifest["sets"][set_name] = builder()
        except (requests.RequestException, ValueError, OSError, zipfile.BadZipFile) as exc:
            failures.append(f"{set_name}: {exc}")
            log.error("%s failed: %s", set_name, exc)
        write_manifest(manifest, args.testsets_dir)
        gc.collect()
    return failures


# ---------------------------------------------------------------------------
# German sets
# ---------------------------------------------------------------------------

def ids_look_shuffled(labels: list[str], trials: int = 2000) -> bool:
    """Do the design labels, read in id order, look like a random order (no blocks, no cycle)?"""
    def runs(sequence: list[str]) -> int:
        return 1 + sum(x != y for x, y in zip(sequence, sequence[1:]))

    generator = rng("german-ids", str(len(labels)))
    shuffled = list(labels)
    counts = []
    for _ in range(trials):
        generator.shuffle(shuffled)
        counts.append(runs(shuffled))
    return abs(runs(labels) - statistics.fmean(counts)) <= 3.5 * statistics.pstdev(counts)


def read_german(path: Path, set_name: str, expected_n: int) -> list[Pair]:
    """The pairs in id order, or ValueError when the file is not ready to convert.

    Source ids are kept, so a German file and pairs_<set>.csv join on pair_id. That is only
    allowed if the ids already say nothing about the relation, so a file whose ids follow
    its design labels is refused. Error messages name ids and counts, never products.
    """
    header, rows = read_csv_rows(path)
    missing = GERMAN_COLUMNS - set(header)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    if len(rows) != expected_n:
        raise ValueError(f"{len(rows)} rows, the pre-registration says {expected_n}")
    rows = sorted(rows, key=lambda row: row["pair_id"] or "")
    if [row["pair_id"] for row in rows] != [f"{set_name}-{i:04d}" for i in range(1, expected_n + 1)]:
        raise ValueError(f"pair ids are not {set_name}-0001 to {set_name}-{expected_n:04d}")
    if not ids_look_shuffled([row["design_label"] or "" for row in rows]):
        raise ValueError("pair ids follow the design labels: shuffle the rows before assigning ids")
    pairs = []
    seen: set[frozenset[str]] = set()
    for row in rows:
        a, b = row["a"] or "", row["b"] or ""
        if not a or not b or a == b:
            raise ValueError(f"{row['pair_id']}: empty or repeated product")
        if frozenset((a, b)) in seen:
            raise ValueError(f"{row['pair_id']}: duplicate unordered pair")
        seen.add(frozenset((a, b)))
        price_a, price_b = ((row[col] or "").strip() for col in ("price_a", "price_b"))
        pairs.append(Pair(a, b, price_a=float(price_a) if price_a else None,
                          price_b=float(price_b) if price_b else None))
    return pairs


def build_german(args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Convert whatever is ready. Returns (converted sets, files not there yet, files rejected)."""
    converted, absent, rejected = [], [], []
    for set_name, (filename, expected_n) in GERMAN_INPUTS.items():
        path = args.german_dir / filename
        if not path.exists():
            absent.append(f"{repo_relative(path)} does not exist yet: --german needs a later run")
            continue
        try:
            pairs = read_german(path, set_name, expected_n)
        except ValueError as exc:
            rejected.append(f"{repo_relative(path)} not converted: {exc}")
            continue
        write_pairs(set_name, pairs, "de", args.testsets_dir, args.blind_dir)
        manifest["sets"][set_name] = {
            "source_url": None, "source_file": repo_relative(path), "file_sha256": sha256_file(path),
            "baskets_after_cleaning": None, "pairs_1to1_before_dedupe": None, "overlap_removed": 0,
            "final_n": len(pairs), "seed": SEED,
        }
        converted.append(set_name)
    write_manifest(manifest, args.testsets_dir)
    return converted, absent, rejected


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(testsets_dir: Path) -> dict[str, Any]:
    path = testsets_dir / "manifest.json"
    sets = {}
    if path.exists():
        sets = json.loads(path.read_text(encoding="utf-8")).get("sets", {})
    return {"seed": SEED, "min_support": MIN_SUPPORT, "min_lift": MIN_LIFT, "gbp_to_eur": GBP_TO_EUR,
            "cap": CAP, "sets": sets}


def write_manifest(manifest: dict[str, Any], testsets_dir: Path) -> None:
    ordered = {name: manifest["sets"][name] for name in SET_ORDER if name in manifest["sets"]}
    text = json.dumps({**manifest, "sets": ordered}, indent=2, ensure_ascii=False) + "\n"
    write_text(testsets_dir / "manifest.json", text)


# ---------------------------------------------------------------------------
# Sanity checks: read back from disk. Messages name ids and counts, never products.
# ---------------------------------------------------------------------------

def check_pairs_file(set_name: str, header: list[str], rows: list[dict[str, str]], problems: list[str]) -> None:
    if header != PAIR_COLUMNS:
        problems.append(f"{set_name}: wrong pairs header")
    ids = [row["pair_id"] for row in rows]
    if ids != [f"{set_name}-{i:04d}" for i in range(1, len(rows) + 1)]:
        problems.append(f"{set_name}: pair ids are not sequential")
    seen: set[frozenset[str]] = set()
    for row in rows:
        if not row["a"] or not row["b"] or row["a"] == row["b"]:
            problems.append(f"{set_name}: {row['pair_id']} has an empty or repeated product")
        key = frozenset((row["a"], row["b"]))
        if key in seen:
            problems.append(f"{set_name}: duplicate unordered pair at {row['pair_id']}")
        seen.add(key)
    lang = "de" if set_name in GERMAN_INPUTS else "en"
    has_metrics = set_name not in GERMAN_INPUTS
    has_prices = set_name != "t2"
    for row in rows:
        if row["set"] != set_name or row["lang_a"] != lang or row["lang_b"] != lang:
            problems.append(f"{set_name}: wrong set or lang at {row['pair_id']}")
        if any(bool(row[col]) != has_metrics for col in ("support", "confidence", "lift")):
            problems.append(f"{set_name}: wrong mining cells at {row['pair_id']}")
        if set_name in ("dev", "t1", "t2") and any(bool(row[col]) != has_prices for col in ("price_a", "price_b")):
            problems.append(f"{set_name}: wrong price cells at {row['pair_id']}")
    if set_name in ("t1", "t2") and len(rows) > CAP:
        problems.append(f"{set_name}: {len(rows)} pairs, over the cap of {CAP}")
    if set_name == "dev" and len(rows) != DEV_EXPECTED_PAIRS:
        problems.append(f"dev: {len(rows)} pairs, expected {DEV_EXPECTED_PAIRS}")


def check_blind_files(set_name: str, rows: list[dict[str, str]], blind_dir: Path, problems: list[str]) -> None:
    truth = {row["pair_id"]: {row["a"], row["b"]} for row in rows}
    orders: dict[str, list[str]] = {}
    firsts: dict[str, dict[str, str]] = {}
    for annotator in ANNOTATORS:
        path = blind_dir / f"{annotator}_{set_name}.csv"
        if not path.exists():
            problems.append(f"{set_name}: {path.name} is missing")
            continue
        header, blind = read_csv_rows(path)
        if header != BLIND_COLUMNS:
            problems.append(f"{set_name}: {path.name} has columns other than {BLIND_COLUMNS}")
        ids = [row["pair_id"] for row in blind]
        if len(ids) != len(truth) or set(ids) != set(truth):
            problems.append(f"{set_name}: {path.name} does not contain exactly the ids in the pairs file")
        for row in blind:
            if {row["product_1"], row["product_2"]} != truth.get(row["pair_id"]):
                problems.append(f"{set_name}: {path.name} has the wrong products at {row['pair_id']}")
        orders[annotator] = ids
        firsts[annotator] = {row["pair_id"]: row["product_1"] for row in blind}
    if len(orders) == 2 and len(truth) > 1:
        if orders["A"] == orders["B"]:
            problems.append(f"{set_name}: A and B have the same row order")
        if firsts["A"] == firsts["B"]:
            problems.append(f"{set_name}: A and B have the same product order in every row")


def check_catalog(set_name: str, rows: list[dict[str, str]], testsets_dir: Path, problems: list[str]) -> None:
    path = testsets_dir / f"catalog_{set_name}.txt"
    if not path.exists():
        problems.append(f"{set_name}: {path.name} is missing")
        return
    names = path.read_text(encoding="utf-8").split("\n")
    if names[-1] != "":
        problems.append(f"{set_name}: {path.name} has no final newline")
    names = names[:-1]
    if len(set(names)) != len(names) or "" in names:
        problems.append(f"{set_name}: {path.name} has blank or repeated names")
    known = set(names)
    for row in rows:
        if row["a"] not in known or row["b"] not in known:
            problems.append(f"{set_name}: {row['pair_id']} names a product missing from {path.name}")


def run_checks(testsets_dir: Path, blind_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Returns (sets checked, problems, warnings)."""
    problems: list[str] = []
    warnings: list[str] = []
    tables: dict[str, list[dict[str, str]]] = {}
    for set_name in SET_ORDER:
        path = testsets_dir / f"pairs_{set_name}.csv"
        if path.exists():
            header, rows = read_csv_rows(path)
            tables[set_name] = rows
            check_pairs_file(set_name, header, rows, problems)
            check_blind_files(set_name, rows, blind_dir, problems)
            if set_name in SETS_WITH_CATALOG:
                check_catalog(set_name, rows, testsets_dir, problems)

    def keys(set_name: str) -> set[frozenset[str]]:
        return {overlap_key(row["a"], row["b"]) for row in tables[set_name]}

    if "t1" in tables and "dev" in tables:
        shared = len(keys("t1") & keys("dev"))
        if shared:
            problems.append(f"t1: {shared} pairs also occur in dev")
    audit = testsets_dir / "t1_dev_audit.csv"
    if "t1" in tables and audit.exists():
        known = {row["pair_id"] for row in tables["t1"]}
        if any(row["pair_id"] not in known for row in read_csv_rows(audit)[1]):
            problems.append("t1: t1_dev_audit.csv names an id that is not in pairs_t1.csv")
    if "t3" in tables and "dev_de" in tables:
        shared = len(keys("t3") & keys("dev_de"))
        if shared:
            warnings.append(f"t3: {shared} pairs also occur in dev_de (kept; not this script's call)")
    return list(tables), problems, warnings


def print_summary(manifest: dict[str, Any]) -> None:
    print(f"{'set':8}{'baskets':>9}{'1->1':>7}{'overlap':>9}{'final':>7}{'catalog':>9}")
    for name in SET_ORDER:
        entry = manifest["sets"].get(name)
        if entry:
            cells = [entry.get(key) for key in ("baskets_after_cleaning", "pairs_1to1_before_dedupe",
                                                "overlap_removed", "final_n", "catalog_n")]
            b, p, o, f, c = ("-" if cell is None else cell for cell in cells)
            print(f"{name:8}{b:>9}{p:>7}{o:>9}{f:>7}{c:>9}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--german", action="store_true",
                        help="only convert data/testsets/german/ into dev_de and t3; no mining")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--testsets-dir", type=Path, default=REPO_ROOT / "data" / "testsets")
    parser.add_argument("--german-dir", type=Path, default=None,
                        help="default: <testsets-dir>/german")
    parser.add_argument("--blind-dir", type=Path, default=REPO_ROOT / "eval" / "gold" / "blind")
    args = parser.parse_args(argv)
    if args.german_dir is None:
        args.german_dir = args.testsets_dir / "german"
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    args.testsets_dir.mkdir(parents=True, exist_ok=True)
    args.blind_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.testsets_dir)

    failures: list[str] = []
    notes: list[str] = []
    if args.german:
        converted, absent, rejected = build_german(args, manifest)
        notes.append("--german converted: " + (", ".join(converted) or "nothing"))
        notes.extend(absent)
        failures.extend(rejected)
    else:
        failures.extend(build_main(args, manifest))
        ready = [f for f, _ in GERMAN_INPUTS.values() if (args.german_dir / f).exists()]
        notes.append(f"German inputs present: {', '.join(ready)}. Convert them with --german."
                     if ready else "German inputs are not there yet: --german needs a later run.")
        t1 = manifest["sets"].get("t1")
        if t1:
            notes.append(f"t1 audit (t1_dev_audit.csv): {t1['pairs_same_stock_codes_as_dev']} pairs have the same "
                         f"stock codes as a dev pair, {t1['pairs_same_text_as_dev_ignoring_punctuation']} the same "
                         f"text ignoring punctuation; {t1['baskets_shared_with_dev']} baskets are in both sheets.")

    checked, problems, warnings = run_checks(args.testsets_dir, args.blind_dir)
    print_summary(manifest)
    print(f"sanity checks on {', '.join(checked) or 'nothing'}: "
          f"{'FAILED' if problems else 'passed'}")
    for message in (*notes, *(f"WARNING {w}" for w in warnings), *(f"PROBLEM {p}" for p in problems),
                    *(f"FAILED {f}" for f in failures)):
        print(message)
    return 1 if problems or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
