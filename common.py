"""Shared helpers for MMCultureQA (SemEval-2027).

Paths come from .env (copy .env.example). Data layout and file formats:
see PROJECT.md, sections 3, 6 and 7.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- paths

REPO = Path(__file__).resolve().parent
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


# ---------------------------------------------------------------- metrics

_normalizer = None


def normalize(text: str) -> str:
    """Whisper EnglishTextNormalizer (lowercase, punctuation, numbers, spelling)."""
    global _normalizer
    if _normalizer is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _normalizer = EnglishTextNormalizer()
    return _normalizer(text or "")


def wer_one(hyp: str, ref: str) -> float:
    """WER for one transcript vs. its reference question (normalized)."""
    import jiwer
    ref_n, hyp_n = normalize(ref), normalize(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return jiwer.wer(ref_n, hyp_n)


def wer(hyps: list[str], refs: list[str]) -> float:
    """Corpus WER with text normalization (Whisper EnglishTextNormalizer)."""
    import jiwer
    if len(hyps) != len(refs):
        raise ValueError(f"len(hyps)={len(hyps)} != len(refs)={len(refs)}")
    pairs = [(normalize(h), normalize(r)) for h, r in zip(hyps, refs)]
    pairs = [(h, r) for h, r in pairs if r]  # jiwer cannot score empty references
    return jiwer.wer([r for _, r in pairs], [h for h, _ in pairs])


# BERTScore settings: fixed here and nowhere else.
# TODO: check against the official scorer before the first submission:
# https://github.com/mmcultureqa-semeval27/mmcultureqa/tree/main/scorer
BERTSCORE_MODEL = "roberta-large"   # bert_score default for lang="en"
BERTSCORE_RESCALE = False           # True spreads scores out, but must match the official setting
BERTSCORE_IDF = False

_scorer = None


def _bertscorer():
    global _scorer
    if _scorer is None:
        import torch
        from bert_score import BERTScorer
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        _scorer = BERTScorer(model_type=BERTSCORE_MODEL, lang=LANG, idf=BERTSCORE_IDF,
                             rescale_with_baseline=BERTSCORE_RESCALE, device=device)
    return _scorer


def bertscore_items(preds: list[str], refs: list[str], batch_size: int = 32) -> list[dict]:
    """Per-item BERTScore: [{"p", "r", "f1"}, ...] in the input order."""
    if len(preds) != len(refs):
        raise ValueError(f"len(preds)={len(preds)} != len(refs)={len(refs)}")
    if not preds:
        return []
    P, R, F = _bertscorer().score([p or "" for p in preds], [r or "" for r in refs],
                                  batch_size=batch_size)
    return [{"p": p, "r": r, "f1": f} for p, r, f in zip(P.tolist(), R.tolist(), F.tolist())]


def bertscore(preds: list[str], refs: list[str]) -> dict:
    """Returns mean {"precision", "recall", "f1"} over all pairs."""
    items = bertscore_items(preds, refs)
    n = len(items) or 1
    return {"precision": sum(x["p"] for x in items) / n,
            "recall": sum(x["r"] for x in items) / n,
            "f1": sum(x["f1"] for x in items) / n}


def evaluate(preds_path: str | Path, split: str = "dev", group_by: str = "category") -> dict:
    """Reads preds, joins them with gold answers (qa split) by id, computes BERTScore
    overall and per `group_by` field (skipped if gold has no such field),
    writes results/<preds name>.json and per-item scores next to the preds file."""
    import datetime as dt

    preds_path = Path(preds_path)
    preds = {r["id"]: r for r in load_jsonl(preds_path)}
    gold = {r["id"]: r for r in load_split("qa", split)}

    ids = [i for i in preds if i in gold]
    missing = [i for i in gold if i not in preds]
    items = bertscore_items([preds[i]["pred"] for i in ids], [gold[i]["answer"] for i in ids])

    res = {
        "preds": preds_path.name,
        "split": split,
        "n": len(ids),
        "n_missing": len(missing),          # gold ids without a prediction
        "bertscore_f1": sum(x["f1"] for x in items) / max(len(items), 1),
        "bertscore_p": sum(x["p"] for x in items) / max(len(items), 1),
        "bertscore_r": sum(x["r"] for x in items) / max(len(items), 1),
        "bertscore_model": BERTSCORE_MODEL,
        "bertscore_rescale": BERTSCORE_RESCALE,
        "date": dt.date.today().isoformat(),
    }

    groups = {}
    for i, x in zip(ids, items):
        g = gold[i].get(group_by)
        if g is not None:
            groups.setdefault(g, []).append(x["f1"])
    if groups:
        res[f"by_{group_by}"] = {g: sum(v) / len(v) for g, v in sorted(groups.items())}

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{preds_path.stem}.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    save_jsonl([{"id": i, **x} for i, x in zip(ids, items)],
               preds_path.with_suffix(".scores.jsonl"))
    return res
