"""Step 06 - cross-corpus lineage (exact + MinHash near-dup clusters).
Outputs:
  data/match/clusters.parquet               h_norm -> cluster id
  results/tables/lineage_exact.csv          share of row-corpus unique items also present in column-corpus (exact)
  results/tables/lineage_near.csv           same, using near-dup clusters (Jaccard >= 0.8)
  results/figures/lineage_near.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import checks_for, dpath, load_cfg, log  # noqa: E402
from minhash import band_keys  # noqa: E402

cfg = load_cfg()
M = cfg["match"]
ck = checks_for(cfg, "06_lineage")
RES = Path(cfg["paths"]["results_root"])
(RES / "tables").mkdir(parents=True, exist_ok=True)
(RES / "figures").mkdir(parents=True, exist_ok=True)

sigs = np.load(dpath(cfg, "match", "sigs.npy"))
order = pd.read_parquet(dpath(cfg, "match", "sigs_order.parquet")).h_norm.tolist()
texts = pd.read_parquet(dpath(cfg, "items", "texts.parquet"), columns=["h_norm", "ntok"])
assert texts.h_norm.tolist() == order, "texts.parquet changed after step 05 - rerun 05"
n = len(order)

# ---------------- union-find over LSH buckets
parent = np.arange(n)


def find(x):
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:
        parent[x], x = r, parent[x]
    return r


valid = texts.ntok.values >= M["shingle_k"]
keys = band_keys(sigs, M["lineage_bands"], M["lineage_rows"])
MAX_BUCKET = 500
merges = 0
for bb in range(M["lineage_bands"]):
    k = keys[:, bb]
    idx = np.nonzero(valid)[0]
    o = idx[np.argsort(k[idx], kind="stable")]
    ks = k[o]
    brk = np.nonzero(np.diff(ks) != 0)[0] + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(o)]])
    big = 0
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        mem = o[s:e]
        if len(mem) > MAX_BUCKET:
            big += 1
            mem = mem[:MAX_BUCKET]
        j = (sigs[mem[1:]] == sigs[mem[0]]).mean(axis=1)
        for m in mem[1:][j >= M["lineage_jaccard"]]:
            ra, rb = find(mem[0]), find(int(m))
            if ra != rb:
                parent[rb] = ra
                merges += 1
    log(f"band {bb}: merges so far={merges}, oversized buckets={big}")

roots = np.array([find(i) for i in range(n)])
clusters = pd.DataFrame({"h_norm": order, "cluster": roots})
clusters.to_parquet(dpath(cfg, "match", "clusters.parquet"), index=False)
ck.info("near-dup clusters", f"{n} unique texts -> {len(np.unique(roots))} clusters ({merges} merges)")

# ---------------- lineage matrices
items = pd.read_parquet(dpath(cfg, "items", "items.parquet"), columns=["corpus", "h_norm"])
items = items[~items.corpus.str.startswith("zz_")].drop_duplicates()
items = items.merge(clusters, on="h_norm")
names = sorted(items.corpus.unique())


def matrix(col):
    sets = {c: set(items.loc[items.corpus == c, col]) for c in names}
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            m.loc[a, b] = len(sets[a] & sets[b]) / max(1, len(sets[a]))
    return m


ex, nr = matrix("h_norm"), matrix("cluster")
ex.to_csv(RES / "tables" / "lineage_exact.csv")
nr.to_csv(RES / "tables" / "lineage_near.csv")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(nr.values.astype(float), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title("Near-duplicate lineage: share of ROW corpus found in COLUMN corpus")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "lineage_near.png", dpi=160)
except Exception as e:
    ck.add("lineage heatmap", False, e, level="WARN")


def g(a, b):
    return float(nr.loc[a, b]) if a in nr.index and b in nr.columns else float("nan")


ck.add("diagonal == 1", bool(np.allclose(np.diag(nr.values.astype(float)), 1.0)), "")
ck.add("Polaris largely inside DeepScaleR (known lineage)", g("polaris_53k", "deepscaler") >= 0.3, f"{g('polaris_53k', 'deepscaler'):.3f}", level="WARN")
ck.add("AceReason overlaps DeepScaleR (known lineage)", g("acereason_math", "deepscaler") >= 0.1, f"{g('acereason_math', 'deepscaler'):.3f}", level="WARN")
ck.add("Nemotron-RL-Math-v2 overlaps DAPO/Skywork (placeholders)", max(g("nemotron_rl_math_v2", "dapo_math_17k"), g("nemotron_rl_math_v2", "skywork_or1_math")) > 0.05,
       f"dapo={g('nemotron_rl_math_v2', 'dapo_math_17k'):.3f} sky={g('nemotron_rl_math_v2', 'skywork_or1_math'):.3f}", level="WARN")
print(nr.round(2).to_string())
ck.save()
