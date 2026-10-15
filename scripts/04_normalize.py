"""Step 04 - normalise every item and build the unique-text table.
Outputs:
  data/items/items.parquet   one row per corpus row: corpus,row_idx,item_uid,h_norm,h_loose,ntok,answer_raw,answer_norm,difficulty
  data/items/texts.parquet   one row per unique normalised text: h_norm,text_clean,norm,h_loose,ntok,answers(json),corpora(json)
  data/bench/bench_norm.parquet
  results/tables/corpus_stats.csv, results/tables/top_prefixes.csv
"""
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from common import (checks_for, dpath, hkey, load_cfg, log, loose_key, norm_answer, norm_text,  # noqa: E402
                    strip_templates, tokens)

cfg = load_cfg()
ck = checks_for(cfg, "04_normalize")
RES = Path(cfg["paths"]["results_root"])
(RES / "tables").mkdir(parents=True, exist_ok=True)


def _proc(args):
    raw, ans = args
    clean = strip_templates(raw or "")
    nm = norm_text(clean)
    lk = loose_key(nm)
    return clean, nm, hkey(nm), hkey(lk) if lk else "", len(tokens(nm)), norm_answer(ans)


def normalise(df: pd.DataFrame, text_col: str, ans_col: str) -> pd.DataFrame:
    with Pool(cfg["runtime"]["workers"]) as pool:
        out = pool.map(_proc, list(zip(df[text_col].tolist(), df[ans_col].tolist())), chunksize=2000)
    o = pd.DataFrame(out, columns=["text_clean", "norm", "h_norm", "h_loose", "ntok", "answer_norm"], index=df.index)
    return pd.concat([df, o], axis=1)


# ---- benchmarks
b = pd.read_parquet(dpath(cfg, "bench", "benchmarks.parquet"))
b = normalise(b, "problem", "answer")
b.to_parquet(dpath(cfg, "bench", "bench_norm.parquet"), index=False)
ck.add("benchmarks normalised", len(b) > 0, f"{len(b)} rows")
ck.add("benchmark texts non-trivial", (b.ntok >= 5).mean() > 0.99, f"min ntok={b.ntok.min()}", level="WARN")

# ---- corpora
raw_dir = Path(cfg["paths"]["data_root"]) / "raw"
files = sorted(raw_dir.glob("*.parquet"))
expected = {n: s.get("expected_unique") for n, s in cfg["corpora"].items()}
missing = [n for n in cfg["corpora"] if not (raw_dir / f"{n}.parquet").exists()]
ck.add("all 16 corpora present", not missing, f"missing={missing}")

items, stats, prefix_rows = [], [], []
for f in files:
    df = pd.read_parquet(f)
    name = df.corpus.iloc[0]
    log(f"normalising {name} ({len(df)} rows)")
    df = normalise(df, "problem_raw", "answer_raw")
    df = df[df.ntok > 0]
    df["item_uid"] = df.corpus + ":" + df.row_idx.astype(str)
    uniq = df.h_norm.nunique()
    exp = expected.get(name)
    stats.append({"corpus": name, "rows": len(df), "unique_norm": uniq, "unique_loose": df.h_loose.nunique(),
                  "expected_unique": exp, "with_answer": int((df.answer_norm != "").sum()),
                  "median_tokens": float(df.ntok.median())})
    if exp:
        ok = abs(uniq - exp) / exp <= 0.10
        ck.add(f"{name} unique within 10% of {exp}", ok, f"{uniq}", level="WARN")
    pref = Counter(" ".join(t.split()[:6]) for t in df.drop_duplicates("h_norm").norm)
    top, cnt = pref.most_common(1)[0]
    share = cnt / max(1, uniq)
    ck.add(f"{name} no dominant leftover template", share < 0.2, f"top prefix {top!r} = {share:.1%}", level="WARN")
    for p, c in pref.most_common(10):
        prefix_rows.append({"corpus": name, "prefix": p, "count": c, "share": c / max(1, uniq)})
    items.append(df)

all_df = pd.concat(items, ignore_index=True)
cols = ["corpus", "row_idx", "item_uid", "h_norm", "h_loose", "ntok", "answer_raw", "answer_norm", "difficulty", "meta"]
all_df[cols].to_parquet(dpath(cfg, "items", "items.parquet"), index=False)

g = all_df.groupby("h_norm")
texts = g.agg(text_clean=("text_clean", "first"), norm=("norm", "first"), h_loose=("h_loose", "first"),
              ntok=("ntok", "first")).reset_index()
ans = g.answer_norm.agg(lambda s: json.dumps(sorted({a for a in s if a})[:5]))
cor = g.corpus.agg(lambda s: json.dumps(sorted(set(s))))
texts = texts.merge(ans.rename("answers"), on="h_norm").merge(cor.rename("corpora"), on="h_norm")
texts.to_parquet(dpath(cfg, "items", "texts.parquet"), index=False)

st = pd.DataFrame(stats)
st.to_csv(RES / "tables" / "corpus_stats.csv", index=False)
pd.DataFrame(prefix_rows).to_csv(RES / "tables" / "top_prefixes.csv", index=False)
real = st[~st.corpus.str.startswith("zz_")]
ck.info("total rows / unique texts", f"{int(real.rows.sum())} rows, {len(texts)} unique texts (all corpora incl. canary)")
print(st.to_string(index=False))
ck.save()
