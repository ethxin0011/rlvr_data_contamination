"""CPU rescoring of saved LoRA adapters (no training, no GPU).

For every benchmark item, and every injected training item, measures
    log P(gold answer | problem, "The final answer is $\\boxed{")
under each run's model. This is continuous, so it detects much smaller shifts
than 0/1 greedy accuracy. Then runs the same near-vs-untouched DiD as analyze.py.

Usage (compute instance, CPU venv):
  python rescore.py --data_dir DATA --adapters_dir ADP --out OUT \
      --runs base_s0 clean_s1 keep_s1 leak_s1
ADP/<run>/adapter_config.json must exist for every run except base_*.
Resumable: finished items are skipped on restart.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_messages  # noqa: E402
from analyze import boot_did, boot_mean  # noqa: E402

PREFIX = "The final answer is $\\boxed{"


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def rows_to_score(data_dir):
    rows = [dict(key=b["bench_id"], kind="bench", problem=b["problem"], answer=str(b["answer"]))
            for b in read_jsonl(data_dir / "bench.jsonl")]
    for arm, kind in [("keep", "false_removal"), ("leak", "exposure")]:
        for r in read_jsonl(data_dir / f"train_{arm}.jsonl"):
            if r["kind"] == kind:
                rows.append(dict(key="train::" + r["item_id"], kind=kind,
                                 problem=r["problem"], answer=str(r["answer"])))
    seen, out = set(), []
    for r in rows:
        if r["key"] not in seen:
            seen.add(r["key"]); out.append(r)
    return out


def encode(tok, problem, answer):
    p = tok.apply_chat_template(build_messages(problem), tokenize=False,
                                add_generation_prompt=True) + PREFIX
    pi = tok(p, add_special_tokens=False).input_ids
    ti = tok(answer.strip() + "}$", add_special_tokens=False).input_ids
    return pi, ti


@torch.no_grad()
def score_run(model, tok, rows, out_path, bs):
    done = {r["key"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["key"] not in done]
    enc = [(r, *encode(tok, r["problem"], r["answer"])) for r in todo]
    enc.sort(key=lambda x: len(x[1]) + len(x[2]))
    log(f"  {len(done)} done, {len(enc)} to score")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.time()
    for i in range(0, len(enc), bs):
        chunk = enc[i:i + bs]
        seqs = [pi + ti for _, pi, ti in chunk]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        recs = []
        for j, (r, pi, ti) in enumerate(chunk):
            a, b = len(pi), len(pi) + len(ti)
            lp = torch.log_softmax(logits[j, a - 1:b - 1].float(), -1)
            s = lp.gather(-1, ids[j, a:b].unsqueeze(-1)).sum().item()
            recs.append({"key": r["key"], "kind": r["kind"], "lp": s, "n_tok": len(ti)})
        with open(out_path, "a") as f:
            f.write("".join(json.dumps(x) + "\n" for x in recs))
        if (i // bs) % 25 == 0:
            n = i + len(chunk)
            log(f"  {n}/{len(enc)}  ETA {(time.time() - t0) / n * (len(enc) - n) / 60:.0f} min")


def analyze(out, runs, data_dir):
    frames = []
    for run in runs:
        d = pd.DataFrame(read_jsonl(out / f"lp_{run}.jsonl")).drop_duplicates("key", keep="last")
        frames.append(d.assign(arm=run.rsplit("_s", 1)[0]))
    df = pd.concat(frames)
    lp = df.groupby(["arm", "key"]).lp.mean().unstack(0)  # seed-averaged
    near_sets = json.load(open(data_dir / "near_sets.json"))
    far = [b for b in near_sets["untouched"] if b in lp.index]
    res = {}
    for arm, kind in [("keep", "false_removal"), ("leak", "exposure")]:
        if arm not in lp or "clean" not in lp:
            continue
        delta = (lp[arm] - lp["clean"]).dropna()
        near = [b for b in near_sets[arm] if b in delta.index]
        am = [b for b in near if near_sets[arm][b]]
        dm = [b for b in near if not near_sets[arm][b]]
        train = [k for k in delta.index if k.startswith("train::")
                 and k in set(df[df.kind == kind].key)]
        res[arm] = {
            "train_items_check": {"n": len(train), **boot_mean(delta[train].values)},
            "did": {"n_near": len(near), "n_far": len(far), **boot_did(delta, near, far)},
            "did_answer_match": {"n": len(am), **boot_did(delta, am, far, seed=1)},
            "did_different_answer": {"n": len(dm), **boot_did(delta, dm, far, seed=2)},
        }
    json.dump(res, open(out / "rescore_results.json", "w"), indent=1, default=float)
    log("RESULTS (log-prob of gold answer, nats; arm minus clean)")
    print(json.dumps(res, indent=1, default=float))

    def sig(x):
        return x.get("lo", np.nan) > 0

    lk, kp = res.get("leak", {}), res.get("keep", {})
    if not lk:
        v = "need clean + leak runs"
    elif not sig(lk["train_items_check"]):
        v = "NO_LEARNING: leak training did not even raise its own training answers -> training too weak."
    elif not sig(lk["did"]):
        v = ("LEARNED_NO_TRANSFER: models absorbed their training answers, but benchmark copies did not "
             "shift near benchmark answers -> no detectable exposure effect at this scale.")
    elif kp and sig(kp["did"]):
        v = ("DETECTABLE + KEEP_SHIFTS: positive control works AND false removals shift near benchmark "
             "answers -> check did_answer_match vs did_different_answer.")
    else:
        v = ("DETECTABLE + KEEP_NULL: positive control works and false removals do not shift near "
             "benchmark answers -> conclusive: false removals carry no answer exposure.")
    log("VERDICT:", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--adapters_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    data_dir, out, adp = Path(a.data_dir), Path(a.out), Path(a.adapters_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = rows_to_score(data_dir)
    log(f"items to score per run: {len(rows)}  | threads {torch.get_num_threads()}")
    for run in a.runs:
        log("RUN", run)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
        if not run.startswith("base"):
            assert (adp / run / "adapter_config.json").exists(), f"missing adapter for {run}"
            model = PeftModel.from_pretrained(model, str(adp / run))
        model.eval()
        score_run(model, tok, rows, out / f"lp_{run}.jsonl", a.bs)
        del model
    analyze(out, a.runs, data_dir)
    log("RESCORE OK ->", out)


if __name__ == "__main__":
    main()
