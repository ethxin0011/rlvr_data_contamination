"""Step 07 (GPU job on t4v3) - Tier 3 semantic nearest neighbours.
Inputs : --texts texts_embed.parquet (h_norm,text_clean)  --bench bench_norm.parquet
Outputs: --out/emb_bench_topk.parquet (bkey,h_norm,cos,rank)
         --out/emb_text_max.parquet  (h_norm,max_cos,best_bkey)
         --out/checks/07_embed.json
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common import Checks, log  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--texts", required=True)
ap.add_argument("--bench", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--topk", type=int, default=50)
ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N texts")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ck = Checks("07_embed", out / "checks")
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck.add("CUDA available", dev == "cuda", torch.cuda.get_device_name(0) if dev == "cuda" else "CPU only")

from sentence_transformers import SentenceTransformer  # noqa: E402

tp = Path(a.texts)
tp = next(tp.glob("*.parquet")) if tp.is_dir() else tp
bp = Path(a.bench)
bp = next(bp.glob("bench_norm.parquet")) if bp.is_dir() else bp
texts = pd.read_parquet(tp, columns=["h_norm", "text_clean"])
bench = pd.read_parquet(bp, columns=["bkey", "text_clean"])
if a.limit:
    texts = texts.head(a.limit)
log(f"{len(texts)} texts, {len(bench)} bench items, model={a.model}")

model = SentenceTransformer(a.model, device=dev, model_kwargs={"torch_dtype": torch.float16} if dev == "cuda" else {})
model.max_seq_length = a.max_len


def enc(lst):
    return model.encode(lst, batch_size=a.batch, normalize_embeddings=True, convert_to_tensor=True,
                        show_progress_bar=False).to(torch.float16)


t0 = time.time()
Eb = enc(bench.text_clean.tolist())                                   # (nb, d)
nb = Eb.shape[0]
k = min(a.topk, len(texts))
best_val = torch.full((nb, k), -2.0, device=Eb.device, dtype=torch.float16)
best_idx = torch.full((nb, k), -1, device=Eb.device, dtype=torch.long)
tmax = np.zeros(len(texts), dtype=np.float32)
targ = np.zeros(len(texts), dtype=np.int64)

order = np.argsort(texts.text_clean.str.len().values)                 # length-sorted for speed
CH = 20000
for s in range(0, len(order), CH):
    ids = order[s:s + CH]
    E = enc(texts.text_clean.values[ids].tolist())
    S = Eb @ E.T                                                      # (nb, ch)
    v, j = S.max(dim=0)
    tmax[ids], targ[ids] = v.float().cpu().numpy(), j.cpu().numpy()
    gid = torch.as_tensor(ids, device=Eb.device).unsqueeze(0).expand(nb, -1)
    cv = torch.cat([best_val, S], dim=1)
    ci = torch.cat([best_idx, gid], dim=1)
    tv, tix = cv.topk(k, dim=1)
    best_val, best_idx = tv, torch.gather(ci, 1, tix)
    if (s // CH) % 10 == 0:
        el = time.time() - t0
        log(f"  {s + len(ids)}/{len(texts)} embedded, {el / 60:.1f} min, ~{(s + len(ids)) / max(el, 1):.0f} texts/s")

bv, bi = best_val.float().cpu().numpy(), best_idx.cpu().numpy()
rows = [{"bkey": bench.bkey.iloc[r], "h_norm": texts.h_norm.iloc[bi[r, c]], "cos": float(bv[r, c]), "rank": c}
        for r in range(nb) for c in range(k) if bi[r, c] >= 0]
pd.DataFrame(rows).to_parquet(out / "emb_bench_topk.parquet", index=False)
pd.DataFrame({"h_norm": texts.h_norm.values, "max_cos": tmax, "best_bkey": bench.bkey.values[targ]}).to_parquet(out / "emb_text_max.parquet", index=False)

ck.add("all texts embedded", int((tmax != 0).sum()) >= len(texts) * 0.999, f"{len(texts)} texts in {(time.time() - t0) / 60:.1f} min")
ck.add("no NaN similarities", bool(np.isfinite(tmax).all()), "")
ck.add("bench self-similarity sane", float((Eb @ Eb.T).diagonal().float().mean()) > 0.99, "")
ck.info("max-cos distribution (p50/p90/p99)", np.percentile(tmax, [50, 90, 99]).round(3).tolist())
ck.save(exit_on_fail=False)
