"""Azure OpenAI (Foundry) client for the LLM judge and the paraphrase canaries.

Minimal-API design: every request sends ONLY `model` + `messages`.
No max tokens, no response_format, no reasoning_effort, no temperature -> works on restrictive deployments,
and the model is never cut off by a token limit. JSON output is enforced by the prompt + a tolerant parser.

Settings come from environment variables; missing ones are read from
/home/azureuser/.contam_secrets.env (KEY=VALUE lines, chmod 600, outside the code folder).

  AZURE_OPENAI_ENDPOINT     https://llmmodelsperf.openai.azure.com
  JUDGE_DEPLOYMENT          gpt-6.1-sol
  AZURE_OPENAI_API_VERSION  2024-12-01-preview
  AZURE_OPENAI_API_KEY      key; leave empty to use Entra ID (needs 'Cognitive Services OpenAI User')
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

SECRETS = Path(os.environ.get("CONTAM_SECRETS", "/home/azureuser/.contam_secrets.env"))
DEFAULTS = {
    "AZURE_OPENAI_ENDPOINT": "https://llmmodelsperf.openai.azure.com",
    "JUDGE_DEPLOYMENT": "gpt-6.1-sol",
    "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
}


def _load_settings() -> None:
    if SECRETS.exists():
        for line in SECRETS.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v and not os.environ.get(k):
                os.environ[k] = v
    for k, v in DEFAULTS.items():
        if not os.environ.get(k):
            os.environ[k] = v


_load_settings()


def _endpoint() -> str:
    """Accept the base endpoint or a full '.../openai/deployments/<dep>/chat/completions?api-version=...' URL."""
    ep = os.environ["AZURE_OPENAI_ENDPOINT"].strip()
    m = re.match(r"(https://[^/]+)(/openai/deployments/([^/?]+))?", ep)
    if m and m.group(3):
        os.environ["JUDGE_DEPLOYMENT"] = m.group(3)
    v = re.search(r"api-version=([\w\-.]+)", ep)
    if v:
        os.environ["AZURE_OPENAI_API_VERSION"] = v.group(1)
    if "/openai/v1" in ep:
        return ep.split("/openai/v1")[0] + "/openai/v1"
    return m.group(1) if m else ep.rstrip("/")


def configured() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("JUDGE_DEPLOYMENT"))


def get_client():
    from openai import AzureOpenAI, OpenAI

    ep = _endpoint()
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    if ep.endswith("/openai/v1"):
        if not key:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            tp = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            return OpenAI(base_url=ep + "/", api_key=tp(), max_retries=0, timeout=600)
        return OpenAI(base_url=ep + "/", api_key=key, max_retries=0, timeout=600)
    ver = os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULTS["AZURE_OPENAI_API_VERSION"])
    if key:
        return AzureOpenAI(azure_endpoint=ep, api_key=key, api_version=ver, max_retries=0, timeout=600)
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    tp = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    return AzureOpenAI(azure_endpoint=ep, azure_ad_token_provider=tp, api_version=ver, max_retries=0, timeout=600)


def _fold_system(messages):
    """Some deployments reject the 'system' role -> merge it into the first user message."""
    sys_txt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if sys_txt and rest:
        rest[0]["content"] = sys_txt + "\n\n" + rest[0]["content"]
    return rest


_NO_SYSTEM = False
_FATAL = ("401", "403", "DeploymentNotFound", "404", "invalid_api_key", "PermissionDenied")


def chat(client, messages, max_tokens=None, json_mode=False, retries=30) -> str:
    """Send only model + messages. `max_tokens` and `json_mode` are accepted for compatibility and ignored.
    Transient errors (429 / 5xx / timeouts / empty replies) are retried with backoff; auth errors fail fast."""
    global _NO_SYSTEM
    dep = os.environ["JUDGE_DEPLOYMENT"]
    last = None
    for i in range(retries):
        msgs = _fold_system(messages) if _NO_SYSTEM else messages
        try:
            r = client.chat.completions.create(model=dep, messages=msgs)
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
            last = f"empty content (finish_reason={r.choices[0].finish_reason})"
        except Exception as e:
            last, msg = e, str(e)
            if "system" in msg.lower() and "role" in msg.lower() and not _NO_SYSTEM:
                _NO_SYSTEM = True
                continue
            if any(f in msg for f in _FATAL):
                raise RuntimeError(f"auth/deployment error (not retried): {msg[:300]}")
            m = re.search(r"retry after (\d+) second", msg, re.I)
            time.sleep(int(m.group(1)) + random.random() if m else min(60, 2 ** min(i, 6) + random.random()))
            continue
        time.sleep(1 + random.random())
    raise RuntimeError(f"LLM call failed after {retries} tries: {last}")


def parse_json(s: str) -> dict:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    lab = re.search(r"\b(SAME|SUBSUMES|VARIANT|DIFFERENT)\b", s.upper())
    return {"label": lab.group(1)} if lab else {}
