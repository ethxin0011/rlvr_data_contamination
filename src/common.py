"""Shared utilities: config, checks, text extraction, normalisation, answer matching, hashing."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import zlib
from fractions import Fraction
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

try:  # fast hashing if available; deterministic fallbacks otherwise
    import xxhash

    def h32(s: str) -> int:
        return xxhash.xxh32_intdigest(s)

    def h64(s: str) -> int:
        return xxhash.xxh64_intdigest(s)
except ImportError:  # pragma: no cover
    def h32(s: str) -> int:
        return zlib.crc32(s.encode("utf-8"))

    def h64(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


# ----------------------------------------------------------------- config / io
def load_cfg(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("CONTAM_CFG") or ROOT / "config.yaml")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("CONTAM_DATA_ROOT"):
        cfg["paths"]["data_root"] = os.environ["CONTAM_DATA_ROOT"]
    if os.environ.get("CONTAM_RESULTS_ROOT"):
        cfg["paths"]["results_root"] = os.environ["CONTAM_RESULTS_ROOT"]
    if os.environ.get("CONTAM_WORKERS"):
        cfg["runtime"]["workers"] = int(os.environ["CONTAM_WORKERS"])
    return cfg


def dpath(cfg: dict, *parts) -> Path:
    p = Path(cfg["paths"]["data_root"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Checks:
    """Collects PASS/WARN/FAIL assertions for a step and writes checks/<step>.json."""

    def __init__(self, step: str, out_dir: str | Path):
        self.step, self.items = step, []
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, ok: bool, detail="", level: str = "FAIL") -> bool:
        status = "PASS" if ok else level
        self.items.append({"check": name, "status": status, "detail": str(detail)})
        print(f"  [{status}] {name} :: {detail}", flush=True)
        return ok

    def info(self, name: str, detail="") -> None:
        self.items.append({"check": name, "status": "INFO", "detail": str(detail)})
        print(f"  [INFO] {name} :: {detail}", flush=True)

    def save(self, exit_on_fail: bool = True) -> int:
        with open(self.out_dir / f"{self.step}.json", "w") as f:
            json.dump(self.items, f, indent=1)
        n = {k: sum(i["status"] == k for i in self.items) for k in ("PASS", "WARN", "FAIL")}
        print(f"=== STEP {self.step}: {n['PASS']} PASS | {n['WARN']} WARN | {n['FAIL']} FAIL ===", flush=True)
        if n["FAIL"] and exit_on_fail:
            sys.exit(2)
        return n["FAIL"]


def checks_for(cfg: dict, step: str) -> Checks:
    return Checks(step, Path(cfg["paths"]["data_root"]) / "checks")


def get_path(obj, dotted: str):
    """Fetch nested value 'a.b.c' from dict-like rows (None if missing)."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, str) and cur.startswith("{"):
            try:
                cur = json.loads(cur)
            except Exception:
                return None
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


# ----------------------------------------------------------------- extraction
PROBLEM_KEYS = ["problem", "question", "Problem", "Question", "prompt", "query", "input",
                "instruction", "messages", "responses_create_params", "conversations"]
ANSWER_KEYS = ["answer", "final_answer", "expected_answer", "ground_truth", "gold", "target",
               "reward_model", "solution"]


def _as_py(v):
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        try:
            return v.tolist()
        except Exception:
            return v
    return v


def text_from(v):
    v = _as_py(v)
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("content", "value", "text", "input", "messages", "prompt", "question"):
            if k in v and v[k] is not None:
                t = text_from(v[k])
                if t and t.strip():
                    return t
        return None
    if isinstance(v, (list, tuple)):
        users = [m for m in v if isinstance(m, dict) and (m.get("role") in ("user", "human") or m.get("from") in ("user", "human"))]
        if users:
            m = users[-1]
            return text_from(m.get("content") if m.get("content") is not None else m.get("value"))
        texts = [t for t in (text_from(x) for x in v) if t and t.strip()]
        return texts[-1] if texts else None
    return str(v)


def extract_problem(row: dict, keys=None):
    for k in keys or PROBLEM_KEYS:
        if k in row and row[k] is not None:
            t = text_from(row[k])
            if t and t.strip():
                return t
    return None


def last_boxed(s: str):
    if not s:
        return None
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j].strip()
    return None


def _answer_scalar(v):
    v = _as_py(v)
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return _answer_scalar(v[0]) if v else None
    if isinstance(v, dict):
        for kk in ("ground_truth", "answer", "value", "expected_answer"):
            if kk in v:
                return _answer_scalar(v[kk])
        return None
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            j = json.loads(s)
            if isinstance(j, list) and j:
                return str(j[0]).strip()
        except Exception:
            pass
    if "####" in s:
        s = s.split("####")[-1].strip()
    return s or None


def extract_answer(row: dict, keys=None):
    for k in keys or ANSWER_KEYS:
        if k not in row or row[k] is None:
            continue
        v = row[k]
        if k == "reward_model":
            v = get_path(_as_py(v), "ground_truth")
        if k == "solution":
            b = last_boxed(str(v))
            if b:
                return b
            continue
        a = _answer_scalar(v)
        if a:
            return a
    return None


_GOOSE_Q = re.compile(r"\*\*Question:\*\*\s*(.*?)\s*\*\*(?:Partial |Masked |Incomplete )?(?:Solution|Answer|Proof|Options|Candidate)[^*]{0,40}:?\*\*", re.S | re.I)


def goose_question(prompt: str):
    """GooseReason prompts embed '**Question:** <problem> **Solution:** <masked solution>'."""
    if not prompt:
        return None
    m = _GOOSE_Q.search(prompt)
    if m:
        return m.group(1).strip()
    i = prompt.find("**Question:**")
    if i >= 0:
        return prompt[i + len("**Question:**"):].strip()[:4000]
    return prompt


# ----------------------------------------------------------------- normalisation
_TEMPLATES = [re.compile(p, re.I | re.S) for p in [
    r"^\s*solve the following math problem step by step\.\s*the last line of your response should be of the form answer:\s*\$answer\s*\(without quotes\)\s*where\s*\$answer is the answer to the problem\.\s*",
    r"\s*remember to put your answer on its own line after \"?answer:\"?\.?\s*$",
    r"^\s*your task is to find the solution to this math problem\.\s*make sure your answer is inside \\boxed\{\}\.\s*",
    r"^\s*please work through this math problem\.\s*enclose your answer in \\boxed\{\}\.\s*",
    r"^\s*solve this problem\.\s*enclose your final answer in \\boxed\{\}\.\s*",
    r"^\s*solve the following math problem\.?\s*(make sure to put the answer \(and only answer\) inside \\boxed\{\}\.)?\s*",
    r"\s*please enclose your (final )?answer in \\boxed\{\}\.?\s*$",
    r"\s*let'?s think step by step and output the final answer within \\boxed\{\}\.?\s*$",
    r"\s*please reason step by step,? and put your final answer within \\boxed\{\}\.?\s*$",
    r"\s*put your final answer (with)?in \\boxed\{\}\.?\s*$",
    r"\s*(make sure to )?put the (final )?answer \(and only answer\) inside \\boxed\{\}\.?\s*$",
    r"\s*return your final response as .{0,200}$",
    r"^\s*(question|problem)\s*:\s*",
    r"^\s*(example|problem|exercise|question|task)?\s*\d+(\.\d+)*\s*[\.\)]\s+",
    r"\s*\(\s*\d+\s*(points?|pts|marks?)\s*\)\s*$",
]]


def strip_templates(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for _ in range(2):
        for p in _TEMPLATES:
            s = p.sub("", s)
    return s.strip()


_LATEX_NOISE = re.compile(r"\\(left|right|displaystyle|textstyle|scriptstyle|bigl|bigr|big|Big|,|;|!|:|quad|qquad)(?![a-zA-Z])")
_LATEX_WRAP = re.compile(r"\\(text|mathrm|mathbf|textbf|textit|operatorname|mathit|emph)\s*\{([^{}]*)\}")
_WS = re.compile(r"\s+")
_TOK = re.compile(r"[a-z]+|\d+|\\[a-z]+|[^\s\w]")


def norm_text(s: str) -> str:
    """Matching normal form (input should already be template-stripped)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = _LATEX_NOISE.sub(" ", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\cdot", "*").replace("\\times", "*")
    s = _LATEX_WRAP.sub(r"\2", s)
    for a in ("$", "\\(", "\\)", "\\[", "\\]"):
        s = s.replace(a, " ")
    s = s.replace("{", " { ").replace("}", " } ")
    return _WS.sub(" ", s).strip()


def loose_key(norm: str) -> str:
    return re.sub(r"[^a-z0-9]", "", norm)


def tokens(norm: str) -> list:
    return _TOK.findall(norm)


def hkey(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


def ngram_hashes(toks: list, n: int) -> set:
    if len(toks) < n:
        return set()
    return {h64("\x1f".join(toks[i:i + n])) for i in range(len(toks) - n + 1)}


# ----------------------------------------------------------------- answers
def norm_answer(a) -> str:
    if a is None:
        return ""
    a = str(a).strip()
    if a.startswith("[") and a.endswith("]"):
        try:
            j = json.loads(a)
            if isinstance(j, list) and j:
                a = str(j[0])
        except Exception:
            pass
    b = last_boxed(a)
    if b:
        a = b
    for x in ("$", "\\!", "\\,", "\\;", "\\left", "\\right", "\\displaystyle"):
        a = a.replace(x, "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\^\{?\\circ\}?", "", a)
    a = re.sub(r"\s+", "", a).rstrip(".").lower()
    m = re.fullmatch(r"-?\\frac\{(-?\d+)\}\{(\d+)\}", a)
    if m:
        a = ("-" if a.startswith("-") else "") + f"{m.group(1)}/{m.group(2)}"
    a = a.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", a) else a
    a = re.sub(r"^(-?)0+(\d)", r"\1\2", a)
    return a


def _num(a: str):
    try:
        if re.fullmatch(r"-?\d+/\d+", a):
            return Fraction(a)
        if re.fullmatch(r"-?\d+(\.\d+)?", a):
            return Fraction(a)
    except Exception:
        return None
    return None


def answers_match(a, b) -> bool:
    na, nb = norm_answer(a), norm_answer(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    x, y = _num(na), _num(nb)
    return x is not None and y is not None and x == y


def is_letter_answer(a: str) -> bool:
    return bool(re.fullmatch(r"\(?[a-j]\)?", (a or "").strip().lower()))
