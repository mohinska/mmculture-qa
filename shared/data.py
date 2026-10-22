"""Paths, jsonl helpers and dataset loading for MMCultureQA (SemEval-2027).

Paths come from .env (copy .env.example).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- paths

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

ROOT = Path(os.environ.get("MMCQA_ROOT") or REPO).expanduser()  # folder with data/ (default: repo)
OUT = Path(os.environ.get("MMCQA_OUT") or ROOT).expanduser()    # where asr/ and preds/ go

DATA = ROOT / "data"
ASR_DIR = OUT / "asr"
PREDS_DIR = OUT / "preds"
RESULTS_DIR = REPO / "results"  # committed to git

LANG = "en"
REGION = "mena"
TASKS = ("qa", "sqa")  # sqa = Task 1 Spoken VQA, qa = Task 2 Textual VQA
SPLITS = ("train", "dev", "devtest", "test")


def split_path(task: str, split: str) -> Path:
    return DATA / task / REGION / f"{split}_{LANG}.jsonl"


def asr_path(split: str, model: str) -> Path:
    return ASR_DIR / f"{split}_{LANG}_{model}.jsonl"


def preds_path(task: str, split: str, model: str, prompt_id: str) -> Path:
    """task: "t1_speech_cascade" | "t1_speech_e2e" | "t2_text"."""
    return PREDS_DIR / f"{task}_{split}_{LANG}_{model}_{prompt_id}.jsonl"


# ---------------------------------------------------------------- jsonl

def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: list[dict], path: str | Path, append: bool = False) -> None:
    """append=True: write after every batch so a crash does not lose results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- data

def _resolve(rel: str) -> Path:
    """jsonl stores media as "images/<id>.jpg"; HF zips unpack under data/archives/."""
    p = Path(rel)
    if p.is_absolute():
        return p
    for base in (DATA, DATA / "archives"):
        if (base / p).exists():
            return base / p
    return DATA / "archives" / p


def load_split(task: str = "qa", split: str = "dev",
               n: int | None = None, seed: int = 0) -> list[dict]:
    """task: "qa" | "sqa". Returns dataset rows with absolute image_path / audio_path.

    n: random subset with a fixed seed. Sampling is done over sorted ids,
    so qa and sqa give the same subset for the same (split, n, seed).
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    path = split_path(task, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Check MMCQA_ROOT in .env")

    rows = load_jsonl(path)
    for i, row in enumerate(rows):
        row["row_idx"] = i
        row.setdefault("lang", LANG)
        row.setdefault("split", split)
        if "image" in row:
            row["image_path"] = str(_resolve(row["image"]))
        if "audio" in row:
            row["audio_path"] = str(_resolve(row["audio"]))

    if n is not None:
        ids = sorted({row["id"] for row in rows})
        if n < len(ids):
            keep = set(random.Random(seed).sample(ids, n))
            rows = [row for row in rows if row["id"] in keep]
    return rows


def check_ids(rows: list[dict]) -> dict:
    """Is `id` a unique key? (It is the image SHA-256, one image may have several questions.)"""
    ids = [row["id"] for row in rows]
    n_unique = len(set(ids))
    return {"rows": len(ids), "unique_ids": n_unique, "duplicates": len(ids) - n_unique}


def check_media(rows: list[dict]) -> dict:
    """Counts missing or empty image/audio files. Returns a few examples of missing paths."""
    report = {}
    for key in ("image_path", "audio_path"):
        paths = [row[key] for row in rows if key in row]
        if not paths:
            continue
        bad = [p for p in paths if not os.path.isfile(p) or os.path.getsize(p) == 0]
        report[key] = {"total": len(paths), "missing": len(bad), "examples": bad[:3]}
    return report
