"""Step 05 - Tier 1 (exact / loose-exact) + Tier 2 (13-gram, 8-gram coverage, MinHash-LSH) vs benchmarks.
Outputs:
  data/match/lexical_pairs.parquet  h_norm,bkey,t1_exact,t1_loose,cov13,cov8,jacc
  data/match/sigs.npy + sigs_order.parquet (MinHash signatures, reused by step 06)
"""
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import checks_for, dpath, load_cfg, log, ngram_hashes, tokens  # noqa: E402
from minhash import band_keys, make_perms, shingles, signature  # noqa: E402

cfg = load_cfg()
M = cfg["match"]
ck = checks_for(cfg, "05_lexical")
A, B = make_perms(M["minhash_perms"])
NL, NS = M["ngram_long"], M["ngram_short"]

bench = pd.read_parquet(dpath(cfg, "bench", "bench_norm.parquet"))
texts = pd.read_parquet(dpath(cfg, "items", "texts.parquet"), columns=["h_norm", "h_loose", "norm", "ntok"])
log(f"{len(texts)} unique texts vs {len(bench)} benchmark items")

# ---------------- Tier 1: exact + loose exact (hash joins)
t1a = texts[["h_norm"]].merge(bench[["h_norm", "bkey"]], on="h_norm")
t1a["t1_exact"] = True
t1b = texts[["h_norm", "h_loose"]].merge(bench[["h_loose", "bkey"]], on="h_loose")[["h_norm", "bkey"]]
t1b["t1_loose"] = True
log(f"T1 exact pairs={len(t1a)}  loose pairs={len(t1b)}")

# ---------------- benchmark indexes (shared with workers via fork)
B_KEYS = bench.bkey.tolist()
IDX_L, IDX_S = defaultdict(list), defaultdict(list)
CNT_L, CNT_S = np.zeros(len(bench)), np.zeros(len(bench))
B_SIGS = np.zeros((len(bench), M["minhash_perms"]), dtype=np.uint64)
for j, nm in enumerate(bench.norm):
    tk = tokens(nm)
    gl, gs = ngram_hashes(tk, NL), ngram_hashes(tk, NS)
    CNT_L[j], CNT_S[j] = len(gl), len(gs)
    for g in gl:
        IDX_L[g].append(j)
    for g in gs:
        IDX_S[g].append(j)
    B_SIGS[j] = signature(shingles(tk, M["shingle_k"]), A, B)
B_BAND = band_keys(B_SIGS, M["bench_bands"], M["bench_rows"])
BAND_MAP = [defaultdict(list) for _ in range(M["bench_bands"])]
for j in range(len(bench)):
    if bench.ntok.iloc[j] >= M["shingle_k"]:
        for bb in range(M["bench_bands"]):
            BAND_MAP[bb][int(B_BAND[j, bb])].append(j)
BAND_SETS = [np.array(list(m.keys()), dtype=np.uint64) for m in BAND_MAP]


def work(chunk):
    h_list, n_list = chunk
    sigs = np.zeros((len(n_list), M["minhash_perms"]), dtype=np.uint64)
    pairs = {}
    for i, nm in enumerate(n_list):
        tk = tokens(nm)
        for idx, cnt, key, n in ((IDX_L, CNT_L, "cov13", NL), (IDX_S, CNT_S, "cov8", NS)):
            hits = defaultdict(int)
            for g in ngram_hashes(tk, n):
                for j in idx.get(g, ()):
                    hits[j] += 1
            for j, c in hits.items():
                cov = c / cnt[j] if cnt[j] else 0.0
                if cov >= (M["keep_cov13"] if key == "cov13" else M["keep_cov8"]):
                    pairs.setdefault((i, j), {})[key] = cov
        sigs[i] = signature(shingles(tk, M["shingle_k"]), A, B)
    keys = band_keys(sigs, M["bench_bands"], M["bench_rows"])
    cand = set()
    for bb in range(M["bench_bands"]):
        rows = np.nonzero(np.isin(keys[:, bb], BAND_SETS[bb]))[0]
        for i in rows:
            for j in BAND_MAP[bb].get(int(keys[i, bb]), ()):
                cand.add((int(i), j))
    for i, j in cand:
        jac = float((sigs[i] == B_SIGS[j]).mean())
        if jac >= M["keep_jacc"]:
            pairs.setdefault((i, j), {})["jacc"] = jac
    out = [{"h_norm": h_list[i], "bkey": B_KEYS[j], **v} for (i, j), v in pairs.items()]
    return sigs, out


CH = 4000
chunks = [(texts.h_norm.iloc[s:s + CH].tolist(), texts.norm.iloc[s:s + CH].tolist()) for s in range(0, len(texts), CH)]
log(f"{len(chunks)} chunks on {cfg['runtime']['workers']} workers")
all_sigs, all_pairs = [], []
with Pool(cfg["runtime"]["workers"]) as pool:
    for k, (sg, pr) in enumerate(pool.imap(work, chunks)):
        all_sigs.append(sg)
        all_pairs.extend(pr)
        if k % 25 == 0:
            log(f"  chunk {k}/{len(chunks)}  pairs so far={len(all_pairs)}")

sigs = np.vstack(all_sigs)
np.save(dpath(cfg, "match", "sigs.npy"), sigs)
texts[["h_norm"]].to_parquet(dpath(cfg, "match", "sigs_order.parquet"), index=False)

lex = pd.DataFrame(all_pairs) if all_pairs else pd.DataFrame(columns=["h_norm", "bkey"])
for c in ("cov13", "cov8", "jacc"):
    if c not in lex:
        lex[c] = np.nan
lex = (lex.merge(t1a[["h_norm", "bkey", "t1_exact"]], on=["h_norm", "bkey"], how="outer")
          .merge(t1b, on=["h_norm", "bkey"], how="outer"))
lex["t1_exact"] = lex.t1_exact.fillna(False).astype(bool)
lex["t1_loose"] = lex.t1_loose.fillna(False).astype(bool) | lex.t1_exact
lex[["cov13", "cov8", "jacc"]] = lex[["cov13", "cov8", "jacc"]].fillna(0.0)
lex.to_parquet(dpath(cfg, "match", "lexical_pairs.parquet"), index=False)
log(f"saved {len(lex)} lexical pairs")

# ---------------- checks with canaries + controls
items = pd.read_parquet(dpath(cfg, "items", "items.parquet"), columns=["corpus", "row_idx", "h_norm"])
key = pd.read_parquet(dpath(cfg, "canary", "canary_key.parquet"))
can = key.merge(items[items.corpus == "zz_canary"], on="row_idx")
hit = can.merge(lex, on=["h_norm", "bkey"], how="left")
hit["lex_hit"] = hit.t1_loose.fillna(False).astype(bool) | (hit.cov13.fillna(0) >= 0.5) | (hit.jacc.fillna(0) >= 0.7)
rec = hit.groupby("ctype").lex_hit.mean().to_dict()
ck.add("canary exact recall == 100% (T1)", rec.get("exact", 0) >= 0.999, rec)
ck.add("canary format recall >= 95%", rec.get("format", 0) >= 0.95, f"{rec.get('format', 0):.3f}")
ck.add("canary numbered recall >= 95%", rec.get("numbered", 0) >= 0.95, f"{rec.get('numbered', 0):.3f}")
ck.info("lexical recall on paraphrase (expected LOW -> motivates semantic tier)", f"{rec.get('paraphrase', float('nan')):.3f}")
ck.info("lexical hits on numchange (negatives; near-dup lexically, judge must reject)", f"{rec.get('numchange', 0):.3f}")
ck.add("signatures aligned", sigs.shape[0] == len(texts), f"{sigs.shape}")
ck.save()
