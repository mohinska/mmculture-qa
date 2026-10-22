"""Metrics for MMCultureQA: WER (ASR) and BERTScore (answers)."""
from __future__ import annotations

import json
from pathlib import Path

from data import LANG, RESULTS_DIR, load_jsonl, load_split, save_jsonl

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
