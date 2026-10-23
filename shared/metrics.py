"""Metrics for MMCultureQA: WER (ASR) and the official scorer (answers)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from data import LANG, RESULTS_DIR, load_jsonl, load_split, require_answers, save_jsonl

# BERTScore, BLEU, ROUGE-L and coverage come from the official scorer, bundled unmodified in
# shared/official_scorer/ (needs requirements.txt, see README). Do not re-implement them here.
OFFICIAL = Path(__file__).resolve().parent / "official_scorer"

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


def _official():
    """Imports the bundled leaderboard scorer lazily, so WER users need no torch."""
    if not (OFFICIAL / "backbone.py").exists():
        raise FileNotFoundError(f"{OFFICIAL}/backbone.py not found: copy the upstream scorer/ there")
    sys.path.insert(0, str(OFFICIAL))
    import backbone
    import score
    score.pin_weights(backbone.HUB_REVISION)   # same weights as the leaderboard
    return backbone


def evaluate(preds_path: str | Path, split: str = "dev", group_by: str = "category") -> dict:
    """Scores a preds file ({"id", "answer", "lang"} rows) against the gold answers of `split` (qa task)
    with the official scorer. Gold is cut to the ids present in the preds file (subsets).

    Writes results/<preds name>.json (the official result plus flat keys) and per-item
    scores next to the preds file (<preds name>.scores.jsonl, fields id, p, r, f1).
    An empty prediction scores 0 everywhere, as on the leaderboard."""
    import datetime as dt

    preds_path = Path(preds_path)
    rows = load_jsonl(preds_path)
    require_answers(rows, preds_path)
    preds = {r["id"]: r for r in rows}
    gold = {r["id"]: r for r in load_split("qa", split)}
    ids = sorted(i for i in preds if i in gold)
    if not ids:
        raise ValueError(f"{preds_path.name}: none of its {len(preds)} ids is in the `{split}` split "
                         "(wrong split?)")
    if any("answer" not in gold[i] for i in ids):
        raise ValueError(f"the `{split}` split has no gold answers; score on dev")
    n_missing = len([i for i in gold if i not in preds])    # gold ids without a prediction

    bb = _official()
    scorer = bb.get_scorer()
    seen, orig = {}, scorer.score

    def spy(cands, refs, **kw):           # the official score() runs once; keep its per-item output
        p, r, f = orig(cands, refs, **kw)
        seen.update(p=p.tolist(), r=r.tolist(), f=f.tolist())
        return p, r, f

    with tempfile.TemporaryDirectory() as tmp:
        gold_path, pred_path = Path(tmp) / "gold.jsonl", Path(tmp) / "prediction.jsonl"
        save_jsonl([{"id": i, "answer": {LANG: gold[i]["answer"]},
                     **{k: gold[i][k] for k in ("country", "category", "subcategory") if gold[i].get(k)}}
                    for i in ids], gold_path)
        save_jsonl([{"id": i, "answer": preds[i].get("answer") or ""} for i in ids], pred_path)
        scorer.score = spy
        try:
            res = bb.evaluate(str(gold_path), str(pred_path), split,
                              submission_metadata=[{"name": bb.LANG_FIELD, "value": LANG}])
        finally:
            scorer.score = orig

    nz = [k for k, i in enumerate(ids) if bb._clean(preds[i].get("answer"))]   # scored items, in id order
    assert len(seen.get("f", [])) == len(nz), "official scorer call did not match the non-empty answers"
    items = [{"id": i, "p": 0.0, "r": 0.0, "f1": 0.0} for i in ids]
    for j, k in enumerate(nz):
        items[k].update(p=seen["p"][j], r=seen["r"][j], f1=seen["f"][j])

    block = res["submission_result"]
    res.update({
        "preds": preds_path.name, "split": split, "n": len(ids), "n_missing": n_missing,
        "bertscore_f1": block[bb.M_F1], "bertscore_p": block[bb.M_P], "bertscore_r": block[bb.M_R],
        "bleu": block[bb.M_BLEU], "rouge_l": block[bb.M_ROUGE], "coverage": block[bb.M_COV],
        "bertscore_model": bb.HUB_MODEL, "bertscore_layer": bb.NUM_LAYERS,
        "bertscore_revision": bb.HUB_REVISION, "date": dt.date.today().isoformat(),
    })
    groups = {}
    for x in items:
        g = gold[x["id"]].get(group_by)
        if g is not None:
            groups.setdefault(g, []).append(x["f1"])
    if groups:
        res[f"by_{group_by}"] = {g: sum(v) / len(v) for g, v in sorted(groups.items())}

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{preds_path.stem}.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    save_jsonl(items, preds_path.with_suffix(".scores.jsonl"))
    return res
