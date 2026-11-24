"""Independent recomputation of H6 and H7 (estimates and paired bootstrap CIs over the 30 bundles,
pooled and per language) and of the Holm-corrected p-values for H2-H5.
Inputs: rating_key.csv, R1_scored.csv, R2_scored.csv (raw), and the raw p-values in summary.json.
Does not import or read experiments/analyze.py or study_stats.py.
Publishable = both raters gave 1. delex/guarded are paired with raw per (bundle, language).
Run: .venv/bin/python experiments/verify/verify_gap.py   (exit 1 if a check fails)"""
import csv, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = json.loads((ROOT / "experiments/results/summary.json").read_text())
LANGS = ("de", "en", "el")
CI_TOL, SEED = 0.02, 7

def rows(p): return list(csv.DictReader(open(p, encoding="utf-8")))
key = {r["rating_id"]: r for r in rows(ROOT / "experiments/results/rating_key.csv")}
r1 = {r["rating_id"]: int(r["publishable"]) for r in rows(ROOT / "eval/gold/blind_rating/R1_scored.csv")}
r2 = {r["rating_id"]: int(r["publishable"]) for r in rows(ROOT / "eval/gold/blind_rating/R2_scored.csv")}
assert set(key) == set(r1) == set(r2) and len(key) == 270
pub = {}
for rid, k in key.items():
    cell = (k["bundle_id"], k["language"], k["condition"])
    assert cell not in pub
    pub[cell] = int(r1[rid] == 1 and r2[rid] == 1)
bundles = sorted({b for b, _, _ in pub})
assert len(bundles) == 30 and all((b, l, c) in pub for b in bundles for l in LANGS for c in ("raw", "guarded", "delex"))

def diff(bs, langs, cond):
    return sum(pub[(b, l, cond)] - pub[(b, l, "raw")] for b in bs for l in langs) / (len(bs) * len(langs))

def boot_ci(langs, cond, n, rng):
    ds = sorted(diff([rng.choice(bundles) for _ in bundles], langs, cond) for _ in range(n))
    return ds[int(0.025 * n)], ds[int(0.975 * n) - 1]

tests = SUMMARY["copy"]["tests"]
out, fails = [], []
def row(name, mine, pubv, tol):
    ok = abs(mine - pubv) <= tol + 1e-12
    out.append(f"| {name} | {mine:+.4f} | {pubv:+.4f} | {abs(mine - pubv):.4f} | {'PASS' if ok else 'FAIL'} |")
    if not ok: fails.append(name)

rng = random.Random(SEED)
for hid, cond in (("H6", "guarded"), ("H7", "delex")):
    for scope, langs in (("pooled", LANGS), ("de", ("de",)), ("en", ("en",)), ("el", ("el",))):
        t = tests[scope][hid]
        est = diff(bundles, langs, cond)
        lo, hi = boot_ci(langs, cond, 10000, rng)
        row(f"{hid} {scope} estimate", est, t["estimate"], 1e-9)
        row(f"{hid} {scope} CI low", lo, t["ci"][0], CI_TOL)
        row(f"{hid} {scope} CI high", hi, t["ci"][1], CI_TOL)
        # the pooled estimate must also match the hypothesis record
        if scope == "pooled":
            h = next(x for x in SUMMARY["hypotheses"] if x["id"] == hid)
            row(f"{hid} pooled estimate (hypotheses record)", est, h["estimate"], 1e-9)

# Holm over H2-H5 from the raw p-values
hyp = {h["id"]: h for h in SUMMARY["hypotheses"]}
fam = ["H2", "H3", "H4", "H5"]
order = sorted(fam, key=lambda h: hyp[h]["p_raw"])
holm, running = {}, 0.0
for i, h in enumerate(order):
    running = max(running, min(1.0, (len(fam) - i) * hyp[h]["p_raw"]))
    holm[h] = running
for h in fam:
    ok = round(holm[h], 4) == round(hyp[h]["p_holm"], 4)
    out.append(f"| {h} Holm p | {holm[h]:.4f} | {hyp[h]['p_holm']:.4f} | {abs(holm[h] - hyp[h]['p_holm']):.4f} | {'PASS' if ok else 'FAIL'} |")
    if not ok: fails.append(h + " Holm")

print("| check | recomputed | summary.json | abs diff | result |\n|---|---|---|---|---|")
print("\n".join(out))
print(f"\n{len(out)} checks, {len(fails)} failed: {fails}")
sys.exit(1 if fails else 0)
