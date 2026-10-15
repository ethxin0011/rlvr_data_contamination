"""Step 03 - calibration canaries (planted leaks with known ground truth).
Writes data/raw/zz_canary.parquet (treated as an extra corpus) + data/canary/canary_key.parquet
Types (positive = should be detected as contamination):
  exact        verbatim copy                                      positive
  format       template wrap + LaTeX/whitespace perturbation       positive
  numbered     'Problem 12.' prefix + '(5 points)' suffix          positive
  paraphrase   LLM rewrite, same math                              positive (needs judge endpoint)
  numchange    every integer changed -> different problem          NEGATIVE control
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

import llm  # noqa: E402
from common import checks_for, dpath, load_cfg, log  # noqa: E402

cfg = load_cfg()
ck = checks_for(cfg, "03_canary")
rng = random.Random(cfg["calibration"]["seed"])
n = cfg["calibration"]["n_per_type"]
bench = pd.read_parquet(dpath(cfg, "bench", "benchmarks.parquet"))
pool = bench[bench.bench.isin(["math500", "gsm8k", "aime24", "aime25", "amc23"])].reset_index(drop=True)


def perturb_format(p: str) -> str:
    q = p.replace("\\frac", "\\dfrac").replace("  ", " ").replace("$", " $ ")
    q = re.sub(r"\s*\n\s*", " ", q)
    return ("Solve the following math problem step by step. The last line of your response should be of the form "
            "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n" + q +
            "\n\nRemember to put your answer on its own line after \"Answer:\"")


def change_numbers(p: str) -> str:
    def rep(m):
        v = int(m.group(0))
        return str(v + rng.randint(2, 7))
    q = re.sub(r"(?<![\\a-zA-Z])\d+", rep, p)
    return q


PARA_PROMPT = ("Rewrite the following math problem using substantially different wording and sentence structure. "
               "Keep every number, condition, and the question mathematically identical so the answer is unchanged. "
               "Do not solve it. Output only the rewritten problem.\n\nPROBLEM:\n{p}")

types = ["exact", "format", "numbered", "numchange"] + (["paraphrase"] if llm.configured() else [])
if "paraphrase" not in types:
    ck.add("paraphrase canaries", False, "judge endpoint not configured -> paraphrase type skipped", level="WARN")
client = llm.get_client() if "paraphrase" in types else None

rows, keys, idx = [], [], 0
for t in types:
    sel = pool.sample(n=min(n, len(pool)), random_state=rng.randint(0, 10**6))
    for _, b in sel.iterrows():
        p = b.problem
        if t == "exact":
            q = p
        elif t == "format":
            q = perturb_format(p)
        elif t == "numbered":
            q = f"Problem {rng.randint(1, 30)}. {p} (5 points)"
        elif t == "numchange":
            q = change_numbers(p)
            if q == p:
                continue
        else:
            try:
                q = llm.chat(client, [{"role": "user", "content": PARA_PROMPT.format(p=p)}], max_tokens=800)
            except Exception as e:
                log(f"paraphrase failed: {e}")
                continue
            if not q or q.strip() == p.strip():
                continue
        rows.append({"corpus": "zz_canary", "row_idx": idx, "problem_raw": q, "answer_raw": "" if t == "numchange" else b.answer,
                     "difficulty": None, "meta": json.dumps({"canary_type": t})})
        keys.append({"row_idx": idx, "ctype": t, "positive": t != "numchange", "bkey": b.bkey})
        idx += 1

pd.DataFrame(rows).to_parquet(dpath(cfg, "raw", "zz_canary.parquet"), index=False)
kdf = pd.DataFrame(keys)
kdf.to_parquet(dpath(cfg, "canary", "canary_key.parquet"), index=False)
counts = kdf.ctype.value_counts().to_dict()
ck.add("canaries built for each type", all(counts.get(t, 0) >= n * 0.8 for t in types), counts)
ex = kdf[kdf.ctype == "paraphrase"].head(1)
if len(ex):
    r = ex.iloc[0]
    log("paraphrase example:\n  ORIG: " + pool.set_index("bkey").loc[r.bkey, "problem"][:300] + "\n  NEW : " + rows[int(r.row_idx)]["problem_raw"][:300])
ck.save()
