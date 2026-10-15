"""Step 10 - final labels, paper tables/figures, calibration, control floor, propagation, release files.
Outputs (results_root):
  tables/contam_bench_side.csv     % of each benchmark found in each corpus (by tier)
  tables/contam_train_side.csv     # contaminated training rows / unique items per corpus
  tables/calibration.csv           recall per tier per canary type + judge negative rejection
  tables/control_floor.csv         sibling-split rates (MATH train vs MATH-500, GSM8K train vs test)
  tables/propagation.csv           per leaked benchmark item: corpora containing it, earliest release
  tables/*.tex                     LaTeX versions
  figures/contam_heatmap.png
  release/labels/<corpus>.parquet  item-level labels (ids + hashes only, no text)
data_root:
  match/contaminated_pairs.parquet (used by step 11)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import checks_for, dpath, load_cfg, log  # noqa: E402

cfg = load_cfg()
F = cfg["final_rule"]
ck = checks_for(cfg, "10_aggregate")
RES = Path(cfg["paths"]["results_root"])
for d in ("tables", "figures", "release/labels"):
    (RES / d).mkdir(parents=True, exist_ok=True)

cand = pd.read_parquet(dpath(cfg, "match", "candidates.parquet"))
cand["pair_id"] = cand.h_norm + "|" + cand.bkey
jp = dpath(cfg, "match", "judge_results.parquet")
if jp.exists():
    jr = pd.read_parquet(jp)[["pair_id", "label", "confidence"]]
    cand = cand.merge(jr, on="pair_id", how="left")
else:
    ck.add("judge results present", False, "no judge_results.parquet -> heuristic rule only", level="WARN")
    cand["label"], cand["confidence"] = np.nan, np.nan

judged = cand.label.notna()
heur = (cand.jacc >= F["jacc"]) | (cand.cov13 >= F["cov13"]) | ((cand.cos >= F["cos_with_answer"]) & cand.answer_match)
cand["contaminated"] = cand.t1_loose | (judged & cand.label.isin(["SAME", "SUBSUMES"])) | (~judged & heur)
cand["variant"] = judged & (cand.label == "VARIANT")
cand["tier"] = np.select(
    [cand.t1_exact, cand.t1_loose, (cand.cov13 >= 0.5) | (cand.jacc >= 0.7)],
    ["T1_exact", "T1_loose", "T2_lexical"], default="T3_semantic")
cp = cand[cand.contaminated][["h_norm", "bkey", "bench", "tier", "cos", "jacc", "cov13", "answer_match", "label"]]
cp.to_parquet(dpath(cfg, "match", "contaminated_pairs.parquet"), index=False)
log(f"{len(cp)} contaminated (text, benchmark) pairs")

items = pd.read_parquet(dpath(cfg, "items", "items.parquet"), columns=["corpus", "row_idx", "item_uid", "h_norm"])
bench = pd.read_parquet(dpath(cfg, "bench", "bench_norm.parquet"), columns=["bkey", "bench"])
nbench = bench.bench.value_counts()
real = items[~items.corpus.str.startswith("zz_")]
ip = real.merge(cp, on="h_norm")                                     # (corpus row) x (benchmark item)

# ---------------- bench side: share of each benchmark present in each corpus
rows = []
tiers = ["T1_exact", "T1_loose", "T2_lexical", "T3_semantic"]
for (c, b), g in ip.groupby(["corpus", "bench"]):
    r = {"corpus": c, "bench": b, "n_bench": int(nbench[b]), "leaked_items": g.bkey.nunique(),
         "leaked_pct": 100 * g.bkey.nunique() / nbench[b]}
    for t in tiers:
        r[f"pct_{t}"] = 100 * g.loc[g.tier == t, "bkey"].nunique() / nbench[b]
    rows.append(r)
bs = pd.DataFrame(rows)
allc = sorted(real.corpus.unique())
grid = pd.MultiIndex.from_product([allc, sorted(nbench.index)], names=["corpus", "bench"]).to_frame(index=False)
bs = grid.merge(bs, on=["corpus", "bench"], how="left").fillna(0)
bs.to_csv(RES / "tables" / "contam_bench_side.csv", index=False)
piv = bs.pivot(index="corpus", columns="bench", values="leaked_pct")
piv.round(1).to_latex(RES / "tables" / "contam_bench_side.tex")
print(piv.round(1).to_string())

# ---------------- train side
tot = real.groupby("corpus").agg(rows=("item_uid", "size"), unique=("h_norm", "nunique"))
con = ip.groupby("corpus").agg(contam_rows=("item_uid", "nunique"), contam_unique=("h_norm", "nunique"))
ts = tot.join(con).fillna(0)
ts["contam_rows_pct"] = 100 * ts.contam_rows / ts.rows
ts["contam_unique_pct"] = 100 * ts.contam_unique / ts.unique
ts.to_csv(RES / "tables" / "contam_train_side.csv")
ts.round(3).to_latex(RES / "tables" / "contam_train_side.tex")

# ---------------- calibration (canaries)
key = pd.read_parquet(dpath(cfg, "canary", "canary_key.parquet"))
can = key.merge(items[items.corpus == "zz_canary"][["row_idx", "h_norm"]], on="row_idx").merge(
    cand, on=["h_norm", "bkey"], how="left")
for c in ("t1_loose", "contaminated"):
    can[c] = can[c].fillna(False).astype(bool)
can["T1"] = can.t1_loose
can["T1+T2"] = can.T1 | (can.cov13.fillna(0) >= 0.5) | (can.jacc.fillna(0) >= 0.7)
can["T1+T2+T3cand"] = can["T1+T2"] | (can.cos.fillna(0) >= cfg["embed"]["cand_cos"])
can["final"] = can.contaminated
cal = can.groupby("ctype")[["T1", "T1+T2", "T1+T2+T3cand", "final"]].mean().round(3)
cal.to_csv(RES / "tables" / "calibration.csv")
print(cal.to_string())
pos_final = can[can.positive].final.mean()
neg_fp = can[~can.positive].final.mean()
ck.add("final recall on positive canaries >= 90%", pos_final >= 0.90, f"{pos_final:.3f}")
ck.add("false-positive rate on numchange negatives <= 10%", neg_fp <= 0.10, f"{neg_fp:.3f}")

# ---------------- control floor (human-written sibling splits)
fl = []
for ctrl, b in cfg["control_pairs"].items():
    g = ip[(ip.corpus == ctrl) & (ip.bench == b)]
    fl.append({"control": ctrl, "bench": b, "bench_items_matched": g.bkey.nunique(), "pct": 100 * g.bkey.nunique() / nbench.get(b, 1),
               "by_tier": json.dumps(g.groupby("tier").bkey.nunique().to_dict())})
pd.DataFrame(fl).to_csv(RES / "tables" / "control_floor.csv", index=False)
ck.info("control floor", fl)

# ---------------- AIME26 (post-dates corpora) sanity
a26 = ip[ip.bench == "aime26"]
ck.add("AIME26 leakage ~0 (released after most corpora)", a26.bkey.nunique() <= 1,
       f"{a26.bkey.nunique()} AIME26 items matched in {sorted(a26.corpus.unique())}", level="WARN")

# ---------------- propagation / lineage of leaks
rel = {n: s.get("release", "") for n, s in cfg["corpora"].items()}
pr = ip.groupby("bkey").agg(n_corpora=("corpus", "nunique"), corpora=("corpus", lambda s: json.dumps(sorted(set(s)))))
pr["earliest_corpus"] = ip.assign(rel=ip.corpus.map(rel)).sort_values("rel").groupby("bkey").corpus.first()
pr.sort_values("n_corpora", ascending=False).to_csv(RES / "tables" / "propagation.csv")
ck.info("leaked benchmark items appearing in >=3 corpora", int((pr.n_corpora >= 3).sum()))

# ---------------- figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(piv.values, cmap="Reds", vmin=0, vmax=max(5, np.nanmax(piv.values)))
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("% of benchmark items found in training corpus (all tiers, judge-confirmed)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "contam_heatmap.png", dpi=160)
except Exception as e:
    ck.add("heatmap", False, e, level="WARN")

# ---------------- release: item-level labels (ids + hashes, no text)
lab = ip.groupby("item_uid").agg(benchmarks=("bkey", lambda s: json.dumps(sorted(set(s)))), tier=("tier", "min"))
for c in allc:
    sub = real[real.corpus == c][["item_uid", "row_idx", "h_norm"]].merge(lab, on="item_uid", how="left")
    sub["contaminated"] = sub.benchmarks.notna()
    sub.to_parquet(RES / "release" / "labels" / f"{c}.parquet", index=False)
ck.add("release label files written", len(list((RES / "release" / "labels").glob("*.parquet"))) == len(allc), f"{len(allc)} corpora")
ck.save()
