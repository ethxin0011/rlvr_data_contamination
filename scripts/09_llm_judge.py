"""Step 09 - Tier 4 LLM judge (Foundry gpt-6.1-sol). Resumable (cache jsonl).
No truncation, no token caps: full benchmark + training texts are sent; the API call is model + messages only.
Usage: python scripts/09_llm_judge.py [--limit 50]
Output: data/match/judge_results.parquet  pair_id,label,confidence,rationale
Labels: SAME | SUBSUMES | VARIANT | DIFFERENT   (SAME/SUBSUMES = contaminated)
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

import llm  # noqa: E402
from common import checks_for, dpath, load_cfg, log  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--retry-errors", action="store_true", help="re-judge cached PARSE_ERROR pairs")
a = ap.parse_args()
cfg = load_cfg()
J = cfg["judge"]
ck = checks_for(cfg, "09_judge" + ("_smoke" if a.limit else ""))
if not llm.configured():
    ck.add("judge env configured", False, "endpoint/deployment missing (see src/llm.py)")
    ck.save()

INSTRUCTIONS = """You audit benchmark contamination in math training data.
Compare a BENCHMARK test problem with a TRAINING item and output one label:
- SAME: the training item poses the same mathematical problem (verbatim, reformatted, reworded, translated, or with extra instructions); solving one solves the other. Differences in numbering, formatting, answer-format instructions, or units notation do not matter.
- SUBSUMES: the training item contains the benchmark problem as a part (e.g. as a sub-question, or embedded together with its solution).
- VARIANT: same scenario/template but numbers, conditions or the asked quantity differ, so the answer is different.
- DIFFERENT: different problem (only topically similar).
Reply with ONLY a JSON object, no markdown, no other text:
{"label": "SAME|SUBSUMES|VARIANT|DIFFERENT", "confidence": 0.0-1.0, "rationale": "<=20 words"}"""

q = pd.read_parquet(dpath(cfg, "match", "judge_queue.parquet"))
if a.limit:
    q = q.head(a.limit)
texts = pd.read_parquet(dpath(cfg, "items", "texts.parquet"), columns=["h_norm", "text_clean", "answers"]).set_index("h_norm")
bench = pd.read_parquet(dpath(cfg, "bench", "bench_norm.parquet"), columns=["bkey", "problem", "answer"]).set_index("bkey")
cache_p = dpath(cfg, "match", "judge_cache.jsonl")
done = {}
if cache_p.exists():
    for line in open(cache_p):
        try:
            r = json.loads(line)
            if a.retry_errors and r.get("label") == "PARSE_ERROR":
                continue
            done[r["pair_id"]] = r
        except Exception:
            pass
todo = [r for r in q.itertuples() if r.pair_id not in done]
log(f"{len(q)} pairs, {len(done)} cached, {len(todo)} to judge, concurrency={J['concurrency']}")
client = llm.get_client()


def judge(r):
    b = bench.loc[r.bkey]
    t = texts.loc[r.h_norm]
    ans = json.loads(t.answers or "[]")
    user = (f"{INSTRUCTIONS}\n\nBENCHMARK problem:\n{b.problem}\nBENCHMARK answer: {b.answer}\n\n"
            f"TRAINING item:\n{t.text_clean}\nTRAINING answer: {ans[0] if ans else '(none)'}")
    lab, j = "PARSE_ERROR", {}
    for _ in range(3):  # re-ask if the reply is not parseable
        j = llm.parse_json(llm.chat(client, [{"role": "user", "content": user}]))
        lab = str(j.get("label", "")).upper().strip()
        if lab in ("SAME", "SUBSUMES", "VARIANT", "DIFFERENT"):
            break
        lab = "PARSE_ERROR"
    return {"pair_id": r.pair_id, "label": lab, "confidence": j.get("confidence"), "rationale": str(j.get("rationale", ""))}


errors = 0
with open(cache_p, "a") as fout, ThreadPoolExecutor(J["concurrency"]) as ex:
    futs = [ex.submit(judge, r) for r in todo]
    for k, f in enumerate(as_completed(futs)):
        try:
            res = f.result()
        except Exception as e:
            errors += 1
            log(f"judge error: {str(e)[:200]}")
            if "auth/deployment" in str(e):
                ex.shutdown(cancel_futures=True)
                break
            continue
        done[res["pair_id"]] = res
        fout.write(json.dumps(res) + "\n")
        if k % 200 == 0:
            fout.flush()
            log(f"  {k}/{len(todo)} judged")

res = pd.DataFrame([done[p] for p in q.pair_id if p in done])
if not a.limit:
    res.to_parquet(dpath(cfg, "match", "judge_results.parquet"), index=False)
cov = len(res) / max(1, len(q))
ck.add("judged coverage >= 99%", cov >= 0.99, f"{len(res)}/{len(q)} (call errors={errors}; re-run to resume)")
ck.add("parse errors <= 2%", (res.label == "PARSE_ERROR").mean() <= 0.02 if len(res) else False,
       res.label.value_counts().to_dict() if len(res) else {})

key = pd.read_parquet(dpath(cfg, "canary", "canary_key.parquet"))
items = pd.read_parquet(dpath(cfg, "items", "items.parquet"), columns=["corpus", "row_idx", "h_norm"])
can = key.merge(items[items.corpus == "zz_canary"], on="row_idx")
can["pair_id"] = can.h_norm + "|" + can.bkey
can = can.merge(res, on="pair_id")
if len(can):
    pos, neg = can[can.positive], can[~can.positive]
    pr = pos.label.isin(["SAME", "SUBSUMES"]).mean() if len(pos) else float("nan")
    nr = (~neg.label.isin(["SAME", "SUBSUMES"])).mean() if len(neg) else float("nan")
    ck.add("judge recall on positive canaries >= 90%", pr >= 0.90, f"{pr:.3f} (n={len(pos)})", level="WARN" if a.limit else "FAIL")
    ck.add("judge rejects numchange negatives >= 80%", nr >= 0.80, f"{nr:.3f} (n={len(neg)})", level="WARN" if a.limit else "FAIL")
else:
    ck.info("canary calibration", "no canary pairs in this batch")
ck.save()
