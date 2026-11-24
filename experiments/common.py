"""Shared paths, file readers and the one Ollama client used by every experiment script.

Paths resolve against common.ROOT at call time, so tests (and a scratch run) can point
the scripts at another tree with SMART_FLYER_ROOT or by patching ROOT.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests

ROOT = Path(os.environ.get("SMART_FLYER_ROOT") or Path(__file__).resolve().parent.parent)

TEST_SETS = ("t1", "t2", "t3")
DEV_SETS = ("dev", "dev_de")
GOLD_LABELS = ("complement", "substitute", "not_a_bundle")
VERDICTS = ("COMPLEMENT", "SUBSTITUTE", "NEITHER", "INVALID")
LABEL_TO_VERDICT = {"complement": "COMPLEMENT", "substitute": "SUBSTITUTE", "not_a_bundle": "NEITHER"}

MODEL = "gemma4:e2b-it-qat"
OLLAMA_URL = "http://localhost:11434"
SEED = 42
CALL_TIMEOUT_S = 180.0
CALL_RETRIES = 3
RETRY_PAUSE_S = 5.0
_LOCK_PATH = Path(tempfile.gettempdir()) / "smart_flyer_ollama.lock"

NUMERIC_COLUMNS = ("support", "confidence", "lift", "price_a", "price_b")


# --- Paths -----------------------------------------------------------------------

def testsets_dir() -> Path:
    return ROOT / "data" / "testsets"


def gold_dir() -> Path:
    return ROOT / "eval" / "gold"


def results_dir() -> Path:
    return ROOT / "experiments" / "results"


def pairs_path(set_id: str) -> Path:
    return testsets_dir() / f"pairs_{set_id}.csv"


def catalog_path(set_id: str) -> Path:
    return testsets_dir() / f"catalog_{set_id}.txt"


def final_gold_path(set_id: str) -> Path:
    return gold_dir() / f"final_{set_id}.csv"


def frozen_pipeline_path() -> Path:
    return ROOT / "experiments" / "frozen" / "export_smart_flyer_v1.py"


def working_pipeline_path() -> Path:
    return ROOT / "pipeline" / "export_smart_flyer.py"


def prompt_selection_path() -> Path:
    return results_dir() / "prompt_selection.json"


def resolve_pipeline_path(given: str | Path | None, set_ids: Iterable[str]) -> Path:
    """The pipeline file to use; test sets may only ever see the frozen copy."""
    frozen = frozen_pipeline_path()
    path = Path(given) if given else (frozen if frozen.exists() else working_pipeline_path())
    touches_test = any(s in TEST_SETS for s in set_ids)
    if touches_test and (not frozen.exists() or path.resolve() != frozen.resolve()):
        sys.exit(f"Refusing to touch a test set without the frozen pipeline ({frozen}); "
                 "the pre-registration requires the freeze first.")
    return path


def winner_variant() -> str:
    path = prompt_selection_path()
    if not path.exists():
        sys.exit(f"{path} is missing: run select_prompt.py first, or pass a variant explicitly.")
    return json.loads(path.read_text(encoding="utf-8"))["winner"]


# --- Readers ---------------------------------------------------------------------

def _to_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def read_pairs(set_id: str) -> list[dict[str, Any]]:
    path = pairs_path(set_id)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist yet")
    rows = read_csv(path)
    for row in rows:
        for col in NUMERIC_COLUMNS:
            row[col] = _to_float(row.get(col))
    return rows


def read_catalog(set_id: str) -> list[str] | None:
    """Catalog names of one set. German sets have none, which means no learned attributes."""
    path = catalog_path(set_id)
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_gold(set_id: str) -> dict[str, str]:
    path = final_gold_path(set_id)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist yet (merge_gold.py finalize writes it)")
    gold = {row["pair_id"]: row["label"] for row in read_csv(path)}
    bad = sorted({label for label in gold.values() if label not in GOLD_LABELS})
    if bad:
        raise ValueError(f"{path}: invalid labels {bad}")
    return gold


def read_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# --- JSONL -----------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """All records of a JSONL file. A torn final line (crash mid-write) is ignored."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"WARNING: ignoring a torn last line in {path}", file=sys.stderr, flush=True)
                continue
            raise
    return records


def latest_by_pair(path: Path) -> dict[str, dict[str, Any]]:
    """pair_id -> its last record, since --retry-errors appends a replacement line."""
    return {rec["pair_id"]: rec for rec in read_jsonl(path)}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Ollama ----------------------------------------------------------------------

@contextmanager
def _ollama_lock() -> Iterator[None]:
    """Only one Ollama call at a time, also between separate scripts on this machine."""
    with open(_LOCK_PATH, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def generate(prompt: str, *, num_predict: int, temperature: float = 0.0, seed: int = SEED,
             timeout: float = CALL_TIMEOUT_S, retries: int = CALL_RETRIES,
             pause: float = RETRY_PAUSE_S) -> dict[str, Any]:
    """One completion. think=False and no JSON mode, because this model needs both.

    Never raises for an Ollama failure: after the retries it returns an empty response
    with an "error" field, and the caller decides what that means for the experiment.
    """
    payload = {
        "model": MODEL, "prompt": prompt, "stream": False, "think": False, "keep_alive": "30m",
        "options": {"temperature": temperature, "seed": seed, "num_predict": num_predict},
    }
    error = ""
    wall_ms = 0.0
    for attempt in range(1, retries + 2):
        with _ollama_lock():
            started = time.perf_counter()
            try:
                reply = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
                reply.raise_for_status()
                data = reply.json()
                return {
                    "response": data.get("response", "") or "",
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "wall_ms": round((time.perf_counter() - started) * 1000, 1),
                    "attempts": attempt,
                    "error": None,
                }
            except (requests.RequestException, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"[:300]
                wall_ms = round((time.perf_counter() - started) * 1000, 1)
        if attempt <= retries:
            time.sleep(pause)
    return {"response": "", "prompt_eval_count": None, "eval_count": None, "total_duration": None,
            "load_duration": None, "wall_ms": wall_ms, "attempts": retries + 1, "error": error}
