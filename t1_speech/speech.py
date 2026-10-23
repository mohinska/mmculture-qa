"""T1_Speech: ASR, cascade inputs and speech-vs-text analysis."""
from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))

from data import (LANG, RESULTS_DIR, asr_path, load_jsonl, load_split,
                  save_jsonl, split_path)
from metrics import normalize, wer, wer_one

ASR_MODELS = {
    "whisper-large-v3-turbo": {"mlx": "mlx-community/whisper-large-v3-turbo",
                               "ct2": "large-v3-turbo",
                               "hf": "openai/whisper-large-v3-turbo"},
    "whisper-large-v3":       {"mlx": "mlx-community/whisper-large-v3-mlx",
                               "ct2": "large-v3",
                               "hf": "openai/whisper-large-v3"},
    "whisper-medium.en":      {"mlx": "mlx-community/whisper-medium.en-mlx",
                               "ct2": "medium.en",
                               "hf": "openai/whisper-medium.en"},
    "whisper-small.en":       {"mlx": "mlx-community/whisper-small.en-mlx",
                               "ct2": "small.en",
                               "hf": "openai/whisper-small.en"},
}
DEFAULT_ASR = "whisper-large-v3-turbo"
ASR_SHORT = {"whisper-large-v3-turbo": "whisperturbo", "whisper-large-v3": "whisperlarge",
             "whisper-medium.en": "whispermedium", "whisper-small.en": "whispersmall"}


def _has(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def default_backend() -> str:
    """mlx on Apple Silicon, hf on a CUDA GPU, ct2 (faster-whisper) on CPU."""
    if sys.platform == "darwin" and platform.machine() == "arm64" and _has("mlx_whisper"):
        return "mlx"
    if _has("torch") and _has("transformers"):
        import torch
        if torch.cuda.is_available():
            return "hf"
    if _has("faster_whisper"):
        return "ct2"
    return "hf"


_hf_pipes: dict = {}


def _hf_pipe(repo: str):
    if repo not in _hf_pipes:
        import torch
        from transformers import pipeline
        cuda = torch.cuda.is_available()
        _hf_pipes[repo] = pipeline(
            "automatic-speech-recognition", model=repo,
            torch_dtype=torch.float16 if cuda else torch.float32,
            device="cuda:0" if cuda else "cpu")
    return _hf_pipes[repo]


_ct2_models: dict = {}


def _ct2_model(repo: str):
    if repo not in _ct2_models:
        from faster_whisper import WhisperModel
        _ct2_models[repo] = WhisperModel(repo, device="cpu", compute_type="int8",
                                         cpu_threads=os.cpu_count() or 4)
    return _ct2_models[repo]


def _load_16k(path: str):
    """Reads a wav file as float32 mono at 16 kHz."""
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = gcd(16000, sr)
        x = resample_poly(x, 16000 // g, sr // g).astype("float32")
    return x


def transcribe(audio_paths: list[str], model: str = DEFAULT_ASR,
               backend: str | None = None, initial_prompt: str | None = None,
               batch_size: int = 16) -> list[str]:
    """Audio files -> English transcripts, same order. Greedy decoding without temperature fallback.

    initial_prompt: context text for the decoder, e.g. a list of place names
    (mlx and ct2 backends only).
    """
    backend = backend or default_backend()
    repo = ASR_MODELS.get(model, {}).get(backend, model)

    if backend == "mlx":
        import mlx_whisper
        out = []
        for p in audio_paths:
            r = mlx_whisper.transcribe(p, path_or_hf_repo=repo, language="en",
                                       initial_prompt=initial_prompt, temperature=0.0, verbose=None)
            out.append(r["text"].strip())
        return out

    if backend == "ct2":
        m = _ct2_model(repo)
        out = []
        for p in audio_paths:
            segments, _ = m.transcribe(_load_16k(p), language="en", beam_size=1, temperature=0.0,
                                       initial_prompt=initial_prompt)
            out.append("".join(s.text for s in segments).strip())
        return out

    if backend == "hf":
        if initial_prompt:
            raise NotImplementedError("initial_prompt: use the mlx or ct2 backend")
        gen = {} if repo.endswith(".en") else {"language": "en", "task": "transcribe"}
        res = _hf_pipe(repo)(list(audio_paths), batch_size=batch_size, generate_kwargs=gen)
        return [r["text"].strip() for r in res]

    raise ValueError(f"backend must be 'mlx', 'ct2' or 'hf', got {backend!r}")


def _key(row: dict) -> tuple:
    return (row["id"], row.get("row_idx"))


def wer_refs(split: str) -> dict:
    """id -> text question from qa (the WER reference)."""
    if not split_path("qa", split).exists():
        return {}
    return {r["id"]: r["question"] for r in load_split("qa", split)}


def run_asr(split: str = "dev", model: str = DEFAULT_ASR, n: int | None = None,
            seed: int = 0, backend: str | None = None, initial_prompt: str | None = None,
            tag: str | None = None, batch_size: int = 16) -> Path:
    """Transcribes the sqa split into asr/{split}_en_{tag or model}.jsonl.

    Resumable: rows already in the file are skipped, results are appended per batch.
    A run with initial_prompt needs its own tag, e.g. "whisper-large-v3-turbo+names".
    """
    if initial_prompt and not tag:
        raise ValueError("initial_prompt changes the experiment: pass a tag, so it gets its own file")
    backend = backend or default_backend()
    rows = load_split("sqa", split, n=n, seed=seed)
    refs = wer_refs(split)
    out = asr_path(split, tag or model)
    done = {_key(r) for r in load_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if _key(r) not in done]
    print(f"{out.name}: {len(rows)} rows, {len(done)} done, {len(todo)} to go")

    t0 = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        texts = transcribe([r["audio_path"] for r in batch], model=model, backend=backend,
                           initial_prompt=initial_prompt, batch_size=batch_size)
        recs = []
        for r, t in zip(batch, texts):
            rec = {"id": r["id"], "row_idx": r["row_idx"], "lang": LANG, "split": split,
                   "model": tag or model, "backend": backend, "transcript": t}
            if r["id"] in refs:
                rec["wer"] = round(wer_one(t, refs[r["id"]]), 4)
            recs.append(rec)
        save_jsonl(recs, out, append=True)
        k = i + len(batch)
        print(f"  {k}/{len(todo)}  {(time.time() - t0) / k:.2f}s per clip", end="\r")
    print()
    return out


def asr_summary(path: str | Path, split: str = "dev") -> dict:
    """Corpus and mean WER, share of perfect transcripts; writes results/asr_<name>.json."""
    path = Path(path)
    recs = load_jsonl(path)
    refs = wer_refs(split)
    recs = [r for r in recs if r["id"] in refs]
    per = [r.get("wer", wer_one(r["transcript"], refs[r["id"]])) for r in recs]
    res = {
        "asr": path.name,
        "split": split,
        "n": len(recs),
        "wer_corpus": round(wer([r["transcript"] for r in recs], [refs[r["id"]] for r in recs]), 4),
        "wer_mean": round(sum(per) / max(len(per), 1), 4),
        "share_wer0": round(sum(w == 0 for w in per) / max(len(per), 1), 4),
        "share_wer_gt_0.25": round(sum(w > 0.25 for w in per) / max(len(per), 1), 4),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"asr_{path.stem}.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def cascade_inputs(asr_file: str | Path, split: str = "dev",
                   n: int | None = None, seed: int = 0) -> list[dict]:
    """qa rows with `question` replaced by the transcript; the original is kept in `question_text`.

    n, seed: same subset as load_split, so the ids match the T2_Text run.
    On test (no qa file) the sqa rows are used and `question_text` is None.
    """
    asr = {r["id"]: r["transcript"] for r in load_jsonl(asr_file)}
    base = load_split("qa" if split_path("qa", split).exists() else "sqa", split, n=n, seed=seed)
    missing = [r["id"] for r in base if r["id"] not in asr]
    if missing:
        print(f"cascade_inputs: {len(missing)} of {len(base)} ids have no transcript and are skipped")
    return [{**r, "question_text": r.get("question"), "question": asr[r["id"]]}
            for r in base if r["id"] in asr]


WER_BINS = [(0.0, 0.0, "0"), (0.0, 0.1, "(0, .10]"), (0.1, 0.25, "(.10, .25]"),
            (0.25, 0.5, "(.25, .50]"), (0.5, float("inf"), "> .50")]


def wer_bin(w: float) -> str:
    if w == 0:
        return WER_BINS[0][2]
    for lo, hi, name in WER_BINS[1:]:
        if lo < w <= hi:
            return name
    return WER_BINS[-1][2]


def join_t1_t2(asr_file, speech_preds, text_preds, split: str = "dev"):
    """One row per id scored in both runs: WER, both answers, both BERTScores, delta.

    Both preds files must have gone through evaluate() (needs *.scores.jsonl).
    """
    import pandas as pd

    def scored(p):
        p = Path(p)
        preds = {r["id"]: r["answer"] for r in load_jsonl(p)}
        scores = {r["id"]: r["f1"] for r in load_jsonl(p.with_suffix(".scores.jsonl"))}
        return preds, scores

    sp, ss = scored(speech_preds)
    tp, ts = scored(text_preds)
    asr = {r["id"]: r for r in load_jsonl(asr_file)}
    gold = {r["id"]: r for r in load_split("qa", split)}

    ids = sorted(set(ss) & set(ts) & set(asr) & set(gold))
    df = pd.DataFrame([{
        "id": i,
        "category": gold[i].get("category"),
        "country": gold[i].get("country"),
        "question": gold[i]["question"],
        "transcript": asr[i]["transcript"],
        "wer": asr[i].get("wer", wer_one(asr[i]["transcript"], gold[i]["question"])),
        "answer": gold[i]["answer"],
        "pred_text": tp[i],
        "pred_speech": sp[i],
        "f1_text": ts[i],
        "f1_speech": ss[i],
    } for i in ids])
    if len(df):
        df["delta"] = df["f1_speech"] - df["f1_text"]
        df["wer_bin"] = df["wer"].map(wer_bin)
    return df


def wer_bin_table(df):
    """Mean BERTScore F1 for text vs speech per WER bin."""
    order = [b[2] for b in WER_BINS]
    t = (df.groupby("wer_bin")
           .agg(n=("id", "size"), f1_text=("f1_text", "mean"),
                f1_speech=("f1_speech", "mean"), delta=("delta", "mean"))
           .reindex([b for b in order if b in set(df["wer_bin"])]))
    t["share"] = t["n"] / t["n"].sum()
    return t.round(4)


_CAP = re.compile(r"\b[A-Z][a-zA-Z'\-]*[a-zA-Z]")


def missed_names(ref: str, hyp: str) -> list[str]:
    """Capitalized words of the reference (except the first word) missing from the transcript."""
    hyp_words = set(normalize(hyp).split())
    out = []
    for m in _CAP.finditer(ref or ""):
        if m.start() == 0:
            continue
        w = normalize(m.group())
        if w and any(x not in hyp_words for x in w.split()):
            out.append(m.group())
    return out
