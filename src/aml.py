"""Azure ML helpers used from the notebook (kernel: Python 3.10 - SDK v2)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Data, Environment
from azure.identity import DefaultAzureCredential

from common import load_cfg

CFG = load_cfg()
AZ = CFG["azure"]
CODE = CFG["paths"]["code_root"]
STATE = Path(CODE) / "logs" / "gpu_state.json"


def client() -> MLClient:
    return MLClient(DefaultAzureCredential(), AZ["subscription_id"], AZ["resource_group"], AZ["workspace"])


def ensure_gpu_cluster(ml: MLClient, name: str | None = None, tier: str = "dedicated", location: str | None = None) -> None:
    name = name or AZ["gpu_cluster"]
    try:
        c = ml.compute.get(name)
        print(f"{name} exists: location={c.location} size={c.size} min={c.min_instances} max={c.max_instances} tier={getattr(c, 'tier', '?')}")
    except Exception:
        kw = dict(name=name, size="Standard_NC4as_T4_v3", min_instances=0, max_instances=1,
                  idle_time_before_scale_down=900, tier=tier)
        if location:
            kw["location"] = location
        ml.compute.begin_create_or_update(AmlCompute(**kw)).result()
        print(f"{name} created ({tier}{', ' + location if location else ''})")


def ensure_env(ml: MLClient) -> str:
    name, ver = AZ["gpu_env_name"], str(AZ["gpu_env_version"])
    try:
        ml.environments.get(name, ver)
        print(f"environment {name}:{ver} exists")
    except Exception:
        cf = Path(CODE) / "envs" / "gpu_conda.yml"
        if cf.exists():
            ml.environments.create_or_update(Environment(name=name, version=ver, image=AZ["gpu_base_image"], conda_file=str(cf)))
            print(f"environment {name}:{ver} registered (conda build, ~10-25 min on first job)")
        else:
            ml.environments.create_or_update(Environment(name=name, version=ver, image=AZ["gpu_base_image"]))
            print(f"environment {name}:{ver} registered (image-only, no conda_file found)")
    return f"azureml:{name}:{ver}"


def upload_folder(ml: MLClient, local: str, name: str) -> str:
    d = ml.data.create_or_update(Data(path=local, type=AssetTypes.URI_FOLDER, name=name, version=time.strftime("%Y%m%d%H%M%S")))
    print(f"uploaded {local} -> azureml:{d.name}:{d.version}")
    return f"azureml:{d.name}:{d.version}"


def submit(ml: MLClient, display: str, cmd: str, inputs: dict, env: str, compute: str | None = None) -> str:
    job = command(code=CODE, command=cmd, environment=env, compute=compute or AZ["gpu_cluster"],
                  inputs={k: Input(type=AssetTypes.URI_FOLDER, path=v) for k, v in inputs.items()},
                  outputs={"out": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")},
                  display_name=display, experiment_name=AZ["experiment"],
                  environment_variables={"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    j = ml.jobs.create_or_update(job)
    print(f"submitted {display}: {j.name}\n  studio: {j.studio_url}")
    save_state(**{display: j.name})
    return j.name


def save_state(**kw) -> dict:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    st = json.load(open(STATE)) if STATE.exists() else {}
    st.update(kw)
    json.dump(st, open(STATE, "w"), indent=1)
    return st


def load_state() -> dict:
    return json.load(open(STATE)) if STATE.exists() else {}


def status(ml: MLClient, names) -> None:
    for n in ([names] if isinstance(names, str) else names):
        j = ml.jobs.get(n)
        print(f"{n:32s} {j.status:12s} {j.display_name}")


def download(ml: MLClient, name: str, local: str) -> Path:
    Path(local).mkdir(parents=True, exist_ok=True)
    ml.jobs.download(name=name, output_name="out", download_path=local)
    p = Path(local)
    print(f"downloaded: {[str(x.relative_to(p)) for x in p.rglob('*') if x.is_file()][:20]}")
    return p