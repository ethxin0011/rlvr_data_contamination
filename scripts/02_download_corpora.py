"""Step 02 - download the 16 RLVR corpora -> data/raw/<corpus>.parquet
columns: corpus, row_idx, problem_raw, answer_raw, difficulty, meta
Usage: python scripts/02_download_corpora.py [--only a,b,c] [--force]

v2: if `datasets` cannot parse a dataset's schema (e.g. "Feature type 'Json' not found" on
nvidia/Nemotron-RL-Math-v2), fall back to downloading the raw parquet/jsonl files with huggingface_hub.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from common import (checks_for, dpath, extract_answer, extract_problem, get_path, goose_question,  # noqa: E402
                    load_cfg, log, text_from)

cfg = load_cfg()
os.environ.setdefault("HF_HOME", cfg["paths"]["hf_cache"])
from datasets import load_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

HEAVY = {"generations", "generated_solution", "completions", "correctness_math_verify", "correctness_llama",
         "finish_reasons", "correctness_count", "r1_solution_1", "r1_solution_2", "r1_solution_3", "messages_full"}
META_COLS = ["data_source", "source", "dataset", "original_dataset", "problem_source", "domain", "question_type", "problem_type"]

ck = checks_for(cfg, "02_corpora" + (("_" + args.only.replace(",", "_")) if args.only else ""))
names = [n for n in cfg["corpora"] if not args.only or n in args.only.split(",")]
names.sort(key=lambda n: cfg["corpora"][n].get("special") == "nemotron_placeholder")  # placeholders last


class FileDS:
    """Minimal stand-in for a HF Dataset, built from raw repo files (schema-agnostic)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_names = list(df.columns)

    def remove_columns(self, cols):
        return FileDS(self.df.drop(columns=[c for c in cols if c in self.df.columns]))

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        for rec in self.df.to_dict("records"):
            yield rec


def load_from_files(spec):
    """Download parquet/jsonl files of a dataset repo directly and concatenate them."""
    from huggingface_hub import HfApi, hf_hub_download
    repo = spec["hf_id"]
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.lower().endswith((".parquet", ".jsonl", ".json")) and not f.startswith(".")
             and "readme" not in f.lower() and "config" not in f.lower()]
    split = spec.get("split", "train")
    pick = [f for f in files if split in f.lower()] or files
    log(f"fallback file loader: {len(pick)} file(s): {pick[:10]}")
    parts = []
    for f in pick:
        p = hf_hub_download(repo, f, repo_type="dataset", cache_dir=cfg["paths"]["hf_cache"])
        parts.append(pd.read_parquet(p) if f.lower().endswith(".parquet") else pd.read_json(p, lines=f.lower().endswith(".jsonl")))
    if not parts:
        raise RuntimeError(f"no parquet/jsonl files found in {repo}")
    return FileDS(pd.concat(parts, ignore_index=True))


def load_corpus(spec):
    try:
        return load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"], cache_dir=cfg["paths"]["hf_cache"])
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "authenticated" in msg.lower():
            raise
        log(f"datasets loader failed ({msg[:120]}) -> trying raw files")
        return load_from_files(spec)


def keep_row(spec, r) -> bool:
    f = spec.get("filter")
    if not f:
        return True
    if f["type"] == "column_regex":
        pat = re.compile(f["pattern"], re.I)
        vals = [str(r[c]) for c in f["cols"] if c in r and r[c] is not None]
        return any(pat.search(v) for v in vals)
    if f["type"] == "numina_checkable":
        ans = str(r.get("answer") or "").strip().lower()
        return (str(r.get("question_type", "")).strip() == "math-word-problem"
                and ans not in ("", "proof", "notfound", "none")
                and str(r.get("problem_is_valid", "Yes")) != "No"
                and str(r.get("solution_is_valid", "Yes")) != "No")
    return True


def find_placeholder(r):
    """Locate a pointer dict {dataset, row, ...} anywhere in the first two levels of a row."""
    for k in ("_hf_question_placeholder", "_hf_placeholder"):
        v = r.get(k)
        if isinstance(v, str) and v.startswith("{"):
            try:
                v = json.loads(v)
            except Exception:
                v = None
        if isinstance(v, dict) and v.get("dataset"):
            return v
    for k, v in r.items():
        if isinstance(v, str) and v.startswith("{") and '"dataset"' in v and '"row"' in v:
            try:
                v = json.loads(v)
            except Exception:
                continue
        if isinstance(v, dict):
            if v.get("dataset") and "row" in v:
                return v
            for v2 in v.values():
                if isinstance(v2, dict) and v2.get("dataset") and "row" in v2:
                    return v2
    return None


def resolve_placeholder(ph: dict):
    """Nemotron NeMo-Gym rows point to {dataset, split, row, prefix, suffix} in DAPO/Skywork rows."""
    hf = ph.get("dataset")
    src = next((n for n, s in cfg["corpora"].items() if s["hf_id"] == hf), None)
    if src is None:
        return None, None, hf
    rp = dpath(cfg, "raw", f"{src}.parquet")
    if not rp.exists():
        return None, None, src
    if src not in _RAW_CACHE:
        _RAW_CACHE[src] = pd.read_parquet(rp, columns=["row_idx", "problem_raw", "answer_raw"]).set_index("row_idx")
    t = _RAW_CACHE[src]
    row = int(ph.get("row", -1))
    if row not in t.index:
        return None, None, src
    rec = t.loc[row]
    return f"{ph.get('prefix') or ''}{rec['problem_raw']}{ph.get('suffix') or ''}", rec["answer_raw"], src


_RAW_CACHE: dict = {}

for name in names:
    spec = cfg["corpora"][name]
    out = dpath(cfg, "raw", f"{name}.parquet")
    if out.exists() and not args.force:
        log(f"{name}: exists, skip (use --force to redo)")
        continue
    log(f"{name}: loading {spec['hf_id']} [{spec.get('config')}] split={spec['split']}")
    try:
        ds = load_corpus(spec)
    except Exception as e:
        ck.add(f"{name} download", False, str(e)[:300])
        continue
    drop = [c for c in ds.column_names if c in HEAVY]
    if drop:
        ds = ds.remove_columns(drop)
    log(f"{name}: {len(ds)} rows, columns={ds.column_names}")
    if spec.get("special") == "nemotron_placeholder":
        first = next(iter(ds))
        log("first row (for debugging): " + json.dumps(first, default=str)[:1200])
    f = spec.get("filter")
    if f and f["type"] == "column_regex" and not any(c in ds.column_names for c in f["cols"]):
        ck.add(f"{name} filter column present", False,
               f"none of {f['cols']} in {ds.column_names}; falling back to answer-bearing short-answer rows", level="WARN")
        spec = dict(spec, filter=None)
        fallback = True
    else:
        fallback = False

    recs, n_kept, n_ph, n_ph_ok = [], 0, 0, 0
    for i, r in enumerate(ds):
        if not keep_row(spec, r):
            continue
        sp = spec.get("special")
        meta = {c: str(r[c])[:80] for c in META_COLS if c in r and r[c] is not None}
        if sp == "goose":
            full = text_from(r.get("question"))
            prob, ans = goose_question(full), extract_answer(r)
            meta["goose_solution_in_prompt"] = True
        elif sp == "nemotron_placeholder":
            prob, ans = extract_problem(r, ["question", "responses_create_params"]), extract_answer(r, ["expected_answer", "answer"])
            ph = find_placeholder(r)
            if (not prob or not prob.strip()) and ph:
                n_ph += 1
                p2, a2, src = resolve_placeholder(ph)
                meta["resolved_from"] = f"{src}:{ph.get('row')}"
                if p2:
                    n_ph_ok += 1
                    prob, ans = p2, (ans or a2)
        else:
            prob, ans = extract_problem(r), extract_answer(r)
        if fallback and (not ans or len(str(ans)) > 64 or "```" in (prob or "")):
            continue
        diff = get_path(r, spec["difficulty_field"]) if spec.get("difficulty_field") else None
        try:
            diff = float(diff) if diff is not None else None
        except Exception:
            diff = None
        recs.append({"corpus": name, "row_idx": i, "problem_raw": prob or "", "answer_raw": "" if ans is None else str(ans),
                     "difficulty": diff, "meta": json.dumps(meta)})
        n_kept += 1
    df = pd.DataFrame(recs)
    if df.empty:
        ck.add(f"{name} rows kept > 0", False, "0 rows kept")
        continue
    nonempty = (df.problem_raw.str.strip().str.len() > 0).mean()
    ans_rate = (df.answer_raw.str.strip().str.len() > 0).mean()
    df.to_parquet(out, index=False)
    ck.add(f"{name} rows kept > 0", n_kept > 0, f"{n_kept} of {len(ds)} rows kept")
    ck.add(f"{name} non-empty problem >= 98%", nonempty >= 0.98, f"{nonempty:.4f}", level="WARN")
    ck.add(f"{name} answer present >= 90%", ans_rate >= 0.90, f"{ans_rate:.4f}", level="WARN")
    if n_ph:
        ck.add(f"{name} placeholders resolved", n_ph_ok == n_ph, f"{n_ph_ok}/{n_ph} resolved (needs DAPO + Skywork raw files)", level="WARN")
    if spec.get("filter") and spec["filter"]["type"] == "column_regex":
        col = next((c for c in spec["filter"]["cols"] if c in ds.column_names), None)
        if col:
            ck.info(f"{name} kept source values", df.meta.map(lambda m: json.loads(m).get(col, "?")).value_counts().head(15).to_dict())
    s = df.sample(min(2, len(df)), random_state=0)
    for _, x in s.iterrows():
        log(f"{name} sample: {x.problem_raw[:150]!r} -> {x.answer_raw[:40]!r}")

ck.save()
