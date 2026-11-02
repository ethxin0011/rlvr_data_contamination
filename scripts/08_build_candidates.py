"""Step 08 - merge all tiers into one candidate table + build the LLM-judge queue.
Inputs : data/match/lexical_pairs.parquet, data/match/emb/{emb_bench_topk,emb_text_max}.parquet
Outputs: data/match/candidates.parquet, data/match/judge_queue.parquet
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import answers_match, checks_for, dpath, load_cfg, log  # noqa: E402

cfg = load_cfg()
E, J = cfg["embed"], cfg["judge"]
ck = checks_for(cfg, "08_candidates")
emb_dir = Path(cfg["paths"]["data_root"]) / "match" / "emb"

lex = pd.read_parquet(dpath(cfg, "match", "lexical_pairs.parquet"))
topk = pd.read_parquet(emb_dir / "emb_bench_topk.parquet")
tmax = pd.read_parquet(emb_dir / "emb_text_max.parquet")
ck.add("embedding outputs present", len(topk) > 0 and len(tmax) > 0, f"topk={len(topk)} tmax={len(tmax)}")

e1 = topk[topk.cos >= E["cand_cos"]][["h_norm", "bkey", "cos"]]
e2 = tmax[tmax.max_cos >= E["train_side_cos"]].rename(columns={"best_bkey": "bkey", "max_cos": "cos"})[["h_norm", "bkey", "cos"]]
emb = pd.concat([e1, e2]).groupby(["h_norm", "bkey"], as_index=False).cos.max()
cand = lex.merge(emb, on=["h_norm", "bkey"], how="outer")
for c in ("cov13", "cov8", "jacc"):
    cand[c] = cand[c].fillna(0.0)
for c in ("t1_exact", "t1_loose"):
    cand[c] = cand[c].fillna(False).astype(bool)
# cos for lexical-only pairs: take the bench top-k value if available
allc = topk[["h_norm", "bkey", "cos"]].rename(columns={"cos": "cos_k"})
cand = cand.merge(allc, on=["h_norm", "bkey"], how="left")
cand["cos"] = cand.cos.fillna(cand.cos_k).fillna(0.0)
cand = cand.drop(columns="cos_k")

texts = pd.read_parquet(dpath(cfg, "items", "texts.parquet"), columns=["h_norm", "answers", "corpora", "ntok"])
bench = pd.read_parquet(dpath(cfg, "bench", "bench_norm.parquet"), columns=["bkey", "bench", "answer_norm"])
cand = cand.merge(texts, on="h_norm", how="left").merge(bench, on="bkey", how="left")
cand["answer_match"] = [any(answers_match(x, b) for x in json.loads(a or "[]")) for a, b in zip(cand.answers, cand.answer_norm)]
cand["score"] = cand[["cos", "jacc", "cov13"]].max(axis=1)
cand.to_parquet(dpath(cfg, "match", "candidates.parquet"), index=False)
log(f"{len(cand)} candidate pairs; t1={int(cand.t1_loose.sum())}")

need = (~cand.t1_loose) & ((cand.cov13 >= 0.3) | (cand.cov8 >= 0.5) | (cand.jacc >= 0.5) | (cand.cos >= E["cand_cos"]))
q = cand[need].sort_values("score", ascending=False)
q = q.groupby("bkey", group_keys=False).head(J["per_bench_cap"])
# always keep canary pairs for calibration even if over the cap
can_h = set(pd.read_parquet(dpath(cfg, "items", "items.parquet"), columns=["corpus", "h_norm"]).query("corpus=='zz_canary'").h_norm)
q = pd.concat([q, cand[need & cand.h_norm.isin(can_h)]]).drop_duplicates(["h_norm", "bkey"])
if len(q) > J["max_pairs"]:
    q = q.sort_values("score", ascending=False).head(J["max_pairs"])
q = q.reset_index(drop=True)
q["pair_id"] = q.h_norm + "|" + q.bkey
q.to_parquet(dpath(cfg, "match", "judge_queue.parquet"), index=False)

ck.add("judge queue non-empty", len(q) > 0, f"{len(q)} pairs")
ck.add("judge queue within budget", len(q) <= J["max_pairs"], f"{len(q)} <= {J['max_pairs']}")
ck.info("queue by benchmark", q.bench.value_counts().to_dict())
ck.info("est. judge tokens", f"~{len(q) * 900 / 1e6:.1f}M input+output tokens")
key = pd.read_parquet(dpath(cfg, "canary", "canary_key.parquet"))
items = pd.read_parquet(dpath(cfg, "items", "items.parquet"), columns=["corpus", "row_idx", "h_norm"])
can = key.merge(items[items.corpus == "zz_canary"], on="row_idx").merge(cand, on=["h_norm", "bkey"], how="left")
can["found_any"] = can.t1_loose.fillna(False).astype(bool) | (can.score.fillna(0) > 0)
r = can.groupby("ctype").found_any.mean().round(3).to_dict()
ck.add("canary positives reach candidate stage >= 95%", all(v >= 0.95 for t, v in r.items() if t != "numchange"), r)
ck.info("canary semantic cos (paraphrase) median", float(np.nanmedian(can.loc[can.ctype == "paraphrase", "cos"])) if (can.ctype == "paraphrase").any() else "n/a")
ck.save()
