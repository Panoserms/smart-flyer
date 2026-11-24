#!/usr/bin/env python3
"""Independent recomputation of the smart-flyer study numbers from raw files.
Does not use analyze.py / export_site_data.py / study_stats.py. Writes VERIFICATION.md next to itself.
Run: python3 experiments/verify/verify_numbers.py   (from the repo root, with the pipeline's Python env)"""
import csv, json, math, hashlib, random, re, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
SETS = ["t1", "t2", "t3"]
SITE = ROOT / "experiments/results/site_data.json"
R = ROOT / "experiments/results"
rows = []  # (check, recomputed, published, status)

def rd(p): return list(csv.DictReader(open(p, newline="", encoding="utf-8")))
def jl(p):
    out = {}
    for l in open(p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); out[r["pair_id"]] = r
    return out
def add(name, rec, pub, tol=5e-4, exact=False):
    if isinstance(rec, (int, float, str, bool)) and isinstance(pub, (int, float, str, bool)) and not isinstance(rec, str):
        ok = (rec == pub) if exact else abs(float(rec) - float(pub)) <= tol
    else:
        ok = rec == pub
    rows.append((name, rec, pub, "PASS" if ok else "FAIL"))
def fmt(x): return f"{x:.4f}" if isinstance(x, float) else str(x)

summ = json.load(open(R / "summary.json"))
site = json.load(open(SITE))

# ---- freeze / t1 integrity
sha = hashlib.sha256((ROOT / "experiments/frozen/export_smart_flyer_v1.py").read_bytes()).hexdigest()
prereg = (ROOT / "experiments/PREREGISTRATION.md").read_text()
m = re.search(r"SHA-256 `([0-9a-f]{64})`", prereg)
add("frozen file SHA-256 vs pre-registration", sha[:16] + "...", (m.group(1) if m else "?")[:16] + "...")
add("frozen SHA vs summary.json meta", sha, summ["meta"]["pipeline_sha256"])
add("frozen SHA vs site meta", sha, site["meta"]["frozen_sha256"])
pairs = {s: list(csv.DictReader(open(ROOT / f"data/testsets/pairs_{s}.csv", encoding="utf-8"))) for s in SETS}
audit = {r["pair_id"] for r in rd(ROOT / "data/testsets/t1_dev_audit.csv")}
add("t1 pair count", len(pairs["t1"]), 98, exact=True)
add("t1 ids overlapping audit list", len({p["pair_id"] for p in pairs["t1"]} & audit), 0, exact=True)

# ---- rules
import cl_features as clf, common
pipe = clf.load_pipeline(str(ROOT / "experiments/frozen/export_smart_flyer_v1.py"))
cp = {}
for s in SETS:
    pp = common.read_pairs(s); cat = common.read_catalog(s)
    cp[s] = clf.analyse_pairs(pp, cat, pipe)

gold = {s: {r["pair_id"]: r["label"] for r in rd(ROOT / f"eval/gold/final_{s}.csv")} for s in SETS}
plain = {s: jl(R / f"triage_{s}_plain_examples.jsonl") for s in SETS}
inf = {s: jl(R / f"triage_{s}_informed_examples.jsonl") for s in SETS}

recs = []  # per pair
for s in SETS:
    ids = [p["pair_id"] for p in pairs[s]]
    add(f"{s} gold ids == pair ids", set(ids) == set(gold[s]), True)
    for i in ids:
        rej = bool(cp[s][i]["cl_reject"]); pv = plain[s][i]["verdict"]; iv = inf[s][i]["verdict"]
        g = gold[s][i] == "complement"
        L = pv == "COMPLEMENT"; Lf = iv == "COMPLEMENT"; C = not rej
        recs.append(dict(set=s, id=i, gold=g, glab=gold[s][i], pv=pv, iv=iv, rej=rej,
                         L=L, C=C, CL=C and L, Lf=Lf, CLf=C and Lf))
CONDS = ["L", "C", "CL", "Lf", "CLf"]

def conf(rs, c):
    tp = sum(r[c] and r["gold"] for r in rs); fp = sum(r[c] and not r["gold"] for r in rs)
    fn = sum((not r[c]) and r["gold"] for r in rs); tn = sum((not r[c]) and not r["gold"] for r in rs)
    return tp, fp, fn, tn
def prf(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, (tp + tn) / (tp + fp + fn + tn)
def calls(rs, c): return {"L": len(rs), "Lf": len(rs), "C": 0, "CL": sum(not r["rej"] for r in rs), "CLf": sum(not r["rej"] for r in rs)}[c]

groups = {s: [r for r in recs if r["set"] == s] for s in SETS}; groups["pooled"] = recs
for g, rs in groups.items():
    add(f"{g} n", len(rs), summ["triage"][g]["n"], exact=True)
    for c in CONDS:
        tp, fp, fn, tn = conf(rs, c); p, r, f, a = prf(tp, fp, fn, tn)
        S = summ["triage"][g]["conditions"][c]; W = site["triage"][g][c]
        for k, v in dict(tp=tp, fp=fp, fn=fn, tn=tn, llm_calls=calls(rs, c)).items():
            add(f"{g} {c} {k} (summary)", v, S[k], exact=True); add(f"{g} {c} {k} (site)", v, W[k], exact=True)
        for k, v in dict(precision=p, recall=r, f1=f, accuracy=a).items():
            add(f"{g} {c} {k} (summary)", v, S[k]); add(f"{g} {c} {k} (site)", v, W[k])

# ---- bootstrap
rng = random.Random(12345)
def f1_of(rs, c):
    return prf(*conf(rs, c))[2]
def p_of(rs, c): return prf(*conf(rs, c))[0]
DIFFS = {"CL-L:f1": (f1_of, "CL", "L"), "CL-L:precision": (p_of, "CL", "L"),
         "CL-C:f1": (f1_of, "CL", "C"), "Lf-L:f1": (f1_of, "Lf", "L")}
B = 10000
for g, rs in groups.items():
    n = len(rs); samples = {k: [] for k in DIFFS}
    for _ in range(B):
        bs = [rs[rng.randrange(n)] for _ in range(n)]
        for k, (fn_, a, b) in DIFFS.items():
            samples[k].append(fn_(bs, a) - fn_(bs, b))
    for k, (fn_, a, b) in DIFFS.items():
        est = fn_(rs, a) - fn_(rs, b); ss = sorted(samples[k])
        lo, hi = ss[int(0.025 * B)], ss[int(0.975 * B) - 1]
        S = summ["triage"][g]["diffs"][k]
        add(f"{g} {k} estimate", est, S["estimate"])
        add(f"{g} {k} CI low (+-0.015)", lo, S["ci"][0], tol=0.015)
        add(f"{g} {k} CI high (+-0.015)", hi, S["ci"][1], tol=0.015)

# ---- McNemar + confusion by gold class
def binom_two_sided(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
for g, rs in groups.items():
    ok = lambda r, c: r[c] == r["gold"]
    b = sum(ok(r, "L") and not ok(r, "CL") for r in rs); c = sum(ok(r, "CL") and not ok(r, "L") for r in rs)
    S = summ["triage"][g]["mcnemar_L_vs_CL"]
    add(f"{g} McNemar L_right_CL_wrong", b, S["L_right_CL_wrong"], exact=True)
    add(f"{g} McNemar CL_right_L_wrong", c, S["CL_right_L_wrong"], exact=True)
    add(f"{g} McNemar exact p", binom_two_sided(b, c), S["p"], tol=1e-9)
    for cond in CONDS:
        for cls in ("complement", "substitute", "not_a_bundle"):
            sub = [r for r in rs if r["glab"] == cls]
            e = {"n": len(sub), "predicted_complement": sum(r[cond] for r in sub)}
            e["predicted_other"] = e["n"] - e["predicted_complement"]
            P = summ["triage"][g]["confusion"]["by_condition"][cond][cls]
            add(f"{g} confusion {cond}/{cls}", [e["n"], e["predicted_complement"], e["predicted_other"]],
                [P["n"], P["predicted_complement"], P["predicted_other"]])

# invalid counts
for g, rs in groups.items():
    add(f"{g} invalid plain", sum(r["pv"] == "INVALID" for r in rs), summ["triage"][g]["invalid"]["plain"]["invalid"], exact=True)
    add(f"{g} invalid informed", sum(r["iv"] == "INVALID" for r in rs), summ["triage"][g]["invalid"]["informed"]["invalid"], exact=True)

# ---- robustness
main_p = {r["id"]: r["pv"] for r in recs}; main_i = {r["id"]: r["iv"] for r in recs}
rej_main = {r["id"]: r["rej"] for r in recs}
sw_p = jl(R / "robustness_swap_plain_examples.jsonl"); sw_i = jl(R / "robustness_swap_informed_examples.jsonl")
cs_p = jl(R / "robustness_case_plain_examples.jsonl")
rr = jl(R / "robustness_rules.jsonl")
add("robustness n (swap plain / informed / case / rules)", [len(sw_p), len(sw_i), len(cs_p), len(rr)], [100] * 4)
fl = lambda d, m: sum(d[i]["verdict"] != m[i] for i in d) / len(d)
add("swap flip L", fl(sw_p, main_p), summ["robustness"]["swap_flip_rate"]["L"])
add("swap flip Lf", fl(sw_i, main_i), summ["robustness"]["swap_flip_rate"]["Lf"])
add("case flip L", fl(cs_p, main_p), summ["robustness"]["case_flip_rate"]["L"])
add("swap flip C (rules file)", sum(r["base_reject"] != r["swap_reject"] for r in rr.values()) / len(rr), summ["robustness"]["swap_flip_rate"]["C"])
add("case flip C (rules file)", sum(r["base_reject"] != r["case_reject"] for r in rr.values()) / len(rr), summ["robustness"]["case_flip_rate"]["C"])
add("robustness rules base_reject == my cl_reject", sum(rr[i]["base_reject"] != rej_main[i] for i in rr), 0, exact=True)
bf_L = {i for i in sw_p if sw_p[i]["verdict"] != main_p[i]}; bf_Lf = {i for i in sw_i if sw_i[i]["verdict"] != main_i[i]}
b, c = len(bf_L - bf_Lf), len(bf_Lf - bf_L)
M = summ["robustness"]["mcnemar_swap_L_vs_Lf"]
add("swap McNemar L-only flipped", b, M["L_flipped_Lf_not"], exact=True)
add("swap McNemar Lf-only flipped", c, M["Lf_flipped_L_not"], exact=True)
add("swap McNemar p", binom_two_sided(b, c), M["p"], tol=1e-9)
ids = sorted(sw_p); n = len(ids); est = (len(bf_Lf) - len(bf_L)) / n
add("swap diff Lf-L estimate", est, summ["robustness"]["swap_diff_Lf_minus_L"]["estimate"])
d = [((i in bf_Lf) - (i in bf_L)) for i in ids]; bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
add("swap diff CI low (+-0.015)", bs[int(.025 * B)], summ["robustness"]["swap_diff_Lf_minus_L"]["ci"][0], tol=0.015)
add("swap diff CI high (+-0.015)", bs[int(.975 * B) - 1], summ["robustness"]["swap_diff_Lf_minus_L"]["ci"][1], tol=0.015)
for k in ("swap_flip_rate", "case_flip_rate"):
    add(f"site robustness {k}", site["robustness"][k], summ["robustness"][k])

# ---- agreement
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum(ca[k] * cb[k] for k in ca) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
for ds in site["datasets"]:
    s = ds["id"]
    A = {r["pair_id"]: r["label"] for r in rd(ROOT / f"eval/gold/annotator_a/{s}.csv")}
    Bb = {r["pair_id"]: r["label"] for r in rd(ROOT / f"eval/gold/annotator_b/{s}.csv")}
    ks = sorted(set(A) & set(Bb))
    add(f"{s} annotator pair coverage", [len(A), len(Bb), len(ks)], [ds["n_pairs"]] * 3)
    add(f"{s} kappa 3-class (site)", kappa([A[k] for k in ks], [Bb[k] for k in ks]), ds["kappa_3class"])
    add(f"{s} kappa binary (site)", kappa([A[k] == "complement" for k in ks], [Bb[k] == "complement" for k in ks]), ds["kappa_binary"])
    add(f"{s} gold counts (site)", dict(Counter(gold[s].values())), ds["gold_counts"])

# ---- copy
key = rd(R / "rating_key.csv")
r1 = {r["rating_id"]: r for r in rd(ROOT / "eval/gold/blind_rating/R1_scored.csv")}
r2 = {r["rating_id"]: r for r in rd(ROOT / "eval/gold/blind_rating/R2_scored.csv")}
add("rating coverage key/R1/R2", [len(key), len(r1), len(r2)], [270] * 3)
cell = {}
for k in key:
    pub = r1[k["rating_id"]]["publishable"] == "1" and r2[k["rating_id"]]["publishable"] == "1"
    cell[(k["bundle_id"], k["condition"], k["language"])] = (pub, k["source"], k["set"])
for lang in ["pooled", "de", "en", "el"]:
    for cond in ["raw", "guarded", "delex"]:
        xs = [v for (b, c, l), v in cell.items() if c == cond and (lang == "pooled" or l == lang)]
        S = summ["copy"]["summary"][lang][cond]
        add(f"copy {lang}/{cond} publishable share", sum(v[0] for v in xs) / len(xs), S["publishable_share"])
        add(f"copy {lang}/{cond} template share", sum(v[1] == "template" for v in xs) / len(xs), S["template_share"])
        add(f"copy {lang}/{cond} publishable (site)", sum(v[0] for v in xs) / len(xs), site["copy"]["summary"][lang][cond]["publishable_share"])
pa = [r1[k["rating_id"]]["publishable"] for k in key]; pb = [r2[k["rating_id"]]["publishable"] for k in key]
add("rater kappa on publishable", kappa(pa, pb), summ["copy"]["rater_kappa"]["publishable"]["kappa"])
add("rater percent agreement", 100 * sum(x == y for x, y in zip(pa, pb)) / 270, summ["copy"]["rater_kappa"]["publishable"]["percent_agreement"], tol=0.01)

# ---- site per-pair records reproduce counts
sp = site["pairs"]
add("site pair count", len(sp), 324, exact=True)
sg = {}
for p in sp: sg.setdefault(p["set"], []).append(p)
sg["pooled"] = sp
for g, ps in sg.items():
    for c in CONDS:
        def val(p):
            rej = (p["cl"]["stop_term"] is not None)  # placeholder, replaced below
            return None
    # derive conditions from site record fields
def site_rej(p):
    return p.get("cl_reject", None)
def rej_of(p):
    c = p["cl"]
    return bool(c["stop_term"]) or bool(c["same_head"]) or bool(c["compound"])
mine0 = {r["id"]: r for r in recs}
add("site per-pair reject (stop_term or same_head or compound) vs my cl_reject, mismatches", sum(rej_of(p) != mine0[p["id"]]["rej"] for p in sp), 0, exact=True)
add("site per-pair verdict/gold mismatches vs raw", sum(mine0[p["id"]]["pv"] != p["llm"]["plain"] or mine0[p["id"]]["iv"] != p["llm"]["informed"] or mine0[p["id"]]["glab"] != p["gold"] for p in sp), 0, exact=True)
for g, ps in sg.items():
    rs2 = [dict(gold=p["gold"] == "complement", L=p["llm"]["plain"] == "COMPLEMENT", Lf=p["llm"]["informed"] == "COMPLEMENT",
                C=not rej_of(p), rej=rej_of(p)) for p in ps]
    for r in rs2: r["CL"] = r["C"] and r["L"]; r["CLf"] = r["C"] and r["Lf"]
    for c in CONDS:
        tp, fp, fn, tn = conf(rs2, c); W = site["triage"][g][c]
        add(f"site per-pair records {g} {c} tp/fp/fn/tn", [tp, fp, fn, tn], [W["tp"], W["fp"], W["fn"], W["tn"]])

# ---- spot checks
rnd = random.Random(7)
mine = {r["id"]: r for r in recs}
sample = rnd.sample(sp, 10); bad = 0
for p in sample:
    s = p["set"]; pr = next(x for x in pairs[s] if x["pair_id"] == p["id"])
    bad += (plain[s][p["id"]]["verdict"] != p["llm"]["plain"] or inf[s][p["id"]]["verdict"] != p["llm"]["informed"]
            or pr["a"] != p["a"] or pr["b"] != p["b"] or gold[s][p["id"]] != p["gold"])
add("spot-check 10 random pairs (verdicts, names, gold) vs raw", bad, 0, exact=True)
finals = {}
for l in open(R / "copy_runs.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["kind"] == "final": finals[(r["bundle_id"], r["condition"], r["lang"])] = r
cells = [(b, c, l) for b in site["copy"]["bundles"] for c in ("raw", "guarded", "delex") for l in ("de", "en", "el")]
cs = rnd.sample(cells, 10); bad = 0; detail = []
for b, c, l in cs:
    sc = b["conditions"][c][l]; f = finals.get((b["id"], c, l))
    if f is None: bad += 1; detail.append((b["id"], c, l, "no final record")); continue
    ok = sc["title"] == f["title"] and sc["copy"] == f["copy"]
    kk = cell[(b["id"], c, l)]
    ok = ok and sc["ratings"]["publishable_both"] == kk[0] and sc["source"] == kk[1]
    if not ok: bad += 1; detail.append((b["id"], c, l, "mismatch"))
add("spot-check 10 random copy cells (verbatim text, publishable, source) vs copy_runs/ratings", bad, 0, exact=True)
bad = 0
for b in site["copy"]["bundles"]:
    for c in ("raw", "guarded", "delex"):
        for l in ("de", "en", "el"):
            f = finals.get((b["id"], c, l)); sc = b["conditions"][c][l]
            bad += (f is None or sc["copy"] != f["copy"] or sc["title"] != f["title"]
                    or sc["ratings"]["publishable_both"] != cell[(b["id"], c, l)][0])
add("all 270 site copy cells vs raw (extra)", bad, 0, exact=True)

# ---- REPORT.md headline
rep = (R / "REPORT.md").read_text()
pf = {c: prf(*conf(recs, c))[2] for c in CONDS}
line = f"F1 L {pf['L']:.3f}, C {pf['C']:.3f}, CL {pf['CL']:.3f}, Lf {pf['Lf']:.3f}, CLf {pf['CLf']:.3f}"
add("REPORT.md pooled F1 headline", line in rep, True)
add("REPORT.md call counts (CL vs L)", f"CL used {calls(recs,'CL')} LLM calls against {calls(recs,'L')} for L" in rep, True)

# ---- human check (amendment A3): recomputed from the author's two label files
HUMAN = [ROOT / f"eval/human/human_labels_batch_{i}.csv" for i in (1, 2)]
if "human_check" in summ and all(p.exists() for p in HUMAN):
    hc = summ["human_check"]
    hl = {r["pair_id"]: r["label"] for p in HUMAN for r in rd(p) if r["label"]}
    allgold = {i: l for s in SETS for i, l in gold[s].items()}
    hs = {"pooled": sorted(hl)} | {s: sorted(i for i in hl if i in gold[s]) for s in SETS}
    for g, ks in hs.items():
        H = hc["pooled"] if g == "pooled" else hc["by_set"][g]
        add(f"human check {g} n", len(ks), H["n"], exact=True)
        add(f"human check {g} kappa 3-class", kappa([hl[k] for k in ks], [allgold[k] for k in ks]), H["kappa_3class"])
        add(f"human check {g} kappa binary", kappa([hl[k] == "complement" for k in ks], [allgold[k] == "complement" for k in ks]),
            H["kappa_binary"])
    add("human check disagreements", sum(hl[k] != allgold[k] for k in hl), hc["n_disagreements"], exact=True)
    hrecs = [dict(r, gold=hl[r["id"]] == "complement") for r in recs if r["id"] in hl]
    srecs = [r for r in recs if r["id"] in hl]
    for lab, rs in (("silver", srecs), ("human", hrecs)):
        for c in CONDS:
            p_, _, f_, _ = prf(*conf(rs, c))
            add(f"human check systems {lab} {c} f1", f_, hc["systems"][lab]["f1"][c])
            add(f"human check systems {lab} {c} precision", p_, hc["systems"][lab]["precision"][c])
    add("REPORT.md has the human check section", "## Human check of the labels (amendment A3" in rep, True)

# ---- report
nfail = sum(r[3] == "FAIL" for r in rows)
out = ["# Verification of the smart-flyer study numbers", "",
       f"**{'ALL CHECKS PASS' if not nfail else f'{nfail} CHECK(S) FAIL'}**: {len(rows)} recomputed values compared with summary.json and the site JSON "
       "(exact for counts, 3 decimals for rates, +-0.015 for bootstrap CIs; own seed, 10,000 resamples).", "",
       "Recomputed by `experiments/verify/verify_numbers.py` from the raw gold, JSONL, rating and pair files; rule outcomes by running the frozen pipeline "
       "through `cl_features.analyse_pairs`. The analysis scripts were not read.", "",
       "| check | recomputed | published | result |", "|---|---|---|---|"]
for n_, a, b, st in rows:
    out.append(f"| {n_} | {fmt(a)} | {fmt(b)} | {st} |")
(Path(__file__).parent / "VERIFICATION.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{len(rows)} checks, {nfail} fail")
for r in rows:
    if r[3] != "PASS": print(r)
