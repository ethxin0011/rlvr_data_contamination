"""Pure-python resume helpers (no torch). Used by 12_grpo_train.py and evalcore.py.
Principle (same as the judge's judge_cache.jsonl that survived restarts): every unit of work is
written to a FIXED location the moment it finishes, and on restart finished units are skipped."""
from __future__ import annotations

import json
import os
from pathlib import Path


def valid_checkpoints(ckpt_dir: Path) -> list:
    """Checkpoints newest-first. A checkpoint counts only if its trainer_state.json parses and a
    model/adapter weight file is present (guards against files cut off by a preemption)."""
    out = []
    if not ckpt_dir.exists():
        return out
    for p in ckpt_dir.glob("checkpoint-*"):
        try:
            step = int(p.name.split("-")[-1])
            json.load(open(p / "trainer_state.json"))
            has_w = (p / "adapter_model.safetensors").exists() or (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
            if has_w:
                out.append((step, p))
        except Exception:
            continue
    return [p for _, p in sorted(out, key=lambda t: t[0], reverse=True)]


def adapter_done(out: Path) -> bool:
    """Training finished completely: marker written AFTER the adapter was saved."""
    a = out / "adapter"
    return (out / "TRAIN_DONE").exists() and (a / "adapter_config.json").exists() and (
        (a / "adapter_model.safetensors").exists() or (a / "adapter_model.bin").exists())


def load_done(partial_path: Path) -> tuple[set, list]:
    """Finished eval generations from a jsonl written one batch at a time. Tolerates a truncated last line."""
    done, recs = set(), []
    if partial_path.exists():
        for line in open(partial_path, errors="ignore"):
            try:
                r = json.loads(line)
                k = (r["bkey"], int(r["sample"]))
                if k not in done:
                    done.add(k)
                    recs.append(r)
            except Exception:
                continue
    return done, recs


def append_records(partial_path: Path, recs: list) -> None:
    with open(partial_path, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def remaining_jobs(jobs: list, bkeys: list, done: set) -> list:
    """jobs = [(row_index, sample_index)]; bkeys[row_index] = benchmark key. Returns the not-yet-finished jobs."""
    return [(i, s) for (i, s) in jobs if (bkeys[i], s) not in done]
