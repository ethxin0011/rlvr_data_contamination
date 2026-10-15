"""Step 00 - environment sanity: packages, CPU/RAM/disk, Hugging Face reachability, judge endpoint."""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import checks_for, load_cfg  # noqa: E402

cfg = load_cfg()
ck = checks_for(cfg, "00_env")

for mod in ["datasets", "pandas", "pyarrow", "numpy", "xxhash", "yaml", "matplotlib", "openai", "huggingface_hub"]:
    try:
        __import__(mod)
        ck.add(f"import {mod}", True)
    except Exception as e:
        ck.add(f"import {mod}", False, e)

ncpu = os.cpu_count()
ck.add("cpu count >= 16", ncpu >= 16, f"{ncpu} vCPU (expect 48 on ralir-v3d-control)", level="WARN")
try:
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    ck.add("RAM >= 128 GB", mem_gb >= 128, f"{mem_gb:.0f} GB (expect ~384)", level="WARN")
except Exception as e:
    ck.add("RAM readable", False, e, level="WARN")

root = Path(cfg["paths"]["data_root"])
root.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(root).free / 1e9
ck.add("free disk at data_root >= 100 GB", free >= 100, f"{free:.0f} GB free at {root}")

try:
    from huggingface_hub import HfApi
    info = HfApi().dataset_info("HuggingFaceH4/MATH-500")
    ck.add("Hugging Face Hub reachable", True, info.id)
except Exception as e:
    ck.add("Hugging Face Hub reachable", False, e)

import llm  # noqa: E402

if llm.configured():
    try:
        out = llm.chat(llm.get_client(), [{"role": "user", "content": 'Reply with JSON {"ok": true}'}], max_tokens=50, json_mode=True)
        ck.add("judge endpoint responds", "ok" in out.lower(), out[:120], level="WARN")
    except Exception as e:
        ck.add("judge endpoint responds", False, e, level="WARN")
else:
    ck.add("judge env vars set (AZURE_OPENAI_ENDPOINT, JUDGE_DEPLOYMENT)", False,
           "not set yet - needed from step 03 (paraphrase canaries) and step 09", level="WARN")

ck.save()
