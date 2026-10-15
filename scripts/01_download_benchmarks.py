"""Step 01 - download test benchmarks -> data/bench/benchmarks.parquet
columns: bench, bench_id, bkey, problem, answer
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from common import checks_for, dpath, extract_answer, extract_problem, load_cfg, log  # noqa: E402

cfg = load_cfg()
os.environ.setdefault("HF_HOME", cfg["paths"]["hf_cache"])
from datasets import load_dataset  # noqa: E402

ck = checks_for(cfg, "01_bench")
ID_KEYS = ["unique_id", "id", "problem_idx", "ID", "idx"]
rows = []

for bench, spec in cfg["benchmarks"].items():
    got = None
    for cand in spec["candidates"]:
        try:
            ds = load_dataset(cand["hf_id"], cand.get("config"), split=cand["split"], cache_dir=cfg["paths"]["hf_cache"])
            got = (cand, ds)
            break
        except Exception as e:
            log(f"{bench}: candidate {cand} failed: {str(e)[:150]}")
    recs = []
    if got:
        cand, ds = got
        for i, r in enumerate(ds):
            p = extract_problem(r)
            a = extract_answer(r)
            bid = next((str(r[k]) for k in ID_KEYS if k in r and r[k] is not None), str(i))
            recs.append({"bench": bench, "bench_id": bid, "problem": p, "answer": a, "source": cand["hf_id"]})
    elif spec.get("manual_file"):
        mf = Path(cfg["paths"]["data_root"]) / spec["manual_file"]
        if mf.exists():
            for i, line in enumerate(open(mf)):
                r = json.loads(line)
                recs.append({"bench": bench, "bench_id": str(r.get("id", i)), "problem": extract_problem(r),
                             "answer": extract_answer(r), "source": str(mf)})
    n = len(recs)
    ok = n == spec["expected"]
    ck.add(f"{bench} count == {spec['expected']}", ok, f"got {n} from {recs[0]['source'] if recs else 'NOTHING'}",
           level="WARN" if spec.get("control") else "FAIL")
    if recs:
        empty_p = sum(not (r["problem"] or "").strip() for r in recs)
        empty_a = sum(not (r["answer"] or "").strip() for r in recs)
        ck.add(f"{bench} no empty problems", empty_p == 0, f"{empty_p} empty")
        ck.add(f"{bench} no empty answers", empty_a == 0, f"{empty_a} empty", level="WARN")
        log(f"{bench} sample: {recs[0]['problem'][:160]!r} -> {recs[0]['answer']!r}")
    rows.extend(recs)

df = pd.DataFrame(rows)
df["bench_id"] = df["bench_id"].astype(str)
dups = df.duplicated(["bench", "bench_id"]).sum()
if dups:  # make ids unique
    df["bench_id"] = df["bench_id"] + "#" + df.groupby(["bench", "bench_id"]).cumcount().astype(str)
df["bkey"] = df["bench"] + "::" + df["bench_id"]
out = dpath(cfg, "bench", "benchmarks.parquet")
df.to_parquet(out, index=False)
ck.info("saved", f"{out} ({len(df)} rows)")
ck.info("per-bench counts", df.bench.value_counts().to_dict())
ck.save()
