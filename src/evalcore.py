"""Batched HF generation + math_verify scoring (T4-friendly), RESUMABLE:
each finished batch is appended to partial_<tag>.jsonl; a restarted job skips what is already there."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from resume import append_records, load_done, remaining_jobs

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def build_prompt(tok, problem: str) -> str:
    msgs = [{"role": "user", "content": problem + SUFFIX}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def score(completion: str, gold: str) -> float:
    from math_verify import parse, verify
    try:
        g = parse("\\boxed{" + str(gold) + "}")
        p = parse(completion)
        return 1.0 if (g and p and verify(g, p)) else 0.0
    except Exception:
        return 0.0


@torch.no_grad()
def evaluate(model, tok, eval_path: str, out_dir: Path, max_new_tokens=1024, samples=4, batch=48, tag="model"):
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = pd.read_json(eval_path, lines=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [build_prompt(tok, p) for p in ev.problem]
    bkeys = ev.bkey.tolist()
    partial = out_dir / f"partial_{tag}.jsonl"
    done, recs = load_done(partial)
    jobs_all = [(i, s) for i in range(len(ev)) for s in range(samples)]
    jobs = remaining_jobs(jobs_all, bkeys, done)
    print(f"[eval {tag}] {len(done)} generations already done, {len(jobs)} remaining", flush=True)
    t0 = time.time()
    model.eval()
    for k in range(0, len(jobs), batch):
        chunk = jobs[k:k + batch]
        enc = tok([prompts[i] for i, _ in chunk], return_tensors="pt", padding=True).to(model.device)
        kw = dict(do_sample=True, temperature=0.6, top_p=0.95) if samples > 1 else dict(do_sample=False)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        new = []
        for (i, s), t in zip(chunk, outs):
            r = ev.iloc[i]
            new.append({"bkey": r.bkey, "bench": r.bench, "group": r.group, "sample": s,
                        "correct": score(t, r.answer), "len": len(t)})
        append_records(partial, new)          # saved to the fixed output folder immediately
        recs.extend(new)
        print(f"[eval {tag}] {len(done) + k + len(chunk)}/{len(jobs_all)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    df = pd.DataFrame(recs)
    df.to_parquet(out_dir / f"preds_{tag}.parquet", index=False)
    summ = df.groupby("group").correct.mean().round(4).to_dict()
    summ.update({f"bench_{b}": v for b, v in df.groupby("bench").correct.mean().round(4).items()})
    json.dump(summ, open(out_dir / f"summary_{tag}.json", "w"), indent=1)
    print("SUMMARY", tag, summ, flush=True)
    return summ
