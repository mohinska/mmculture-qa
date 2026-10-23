"""Task 1 cascade: ASR transcript + image -> the T2_Text VLM -> preds.

Same model, prompt and generation settings as t2_text/run_vlm.py; only the
question comes from asr/ instead of the text file.

Kaggle:  python t1_speech/run_cascade.py --model Qwen/Qwen3-VL-4B-Instruct --n 200
Needs asr/{split}_en_{asr}.jsonl (python t1_speech/run_asr.py --split dev).
Resumable: ids already in the preds file are skipped.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "shared"), str(ROOT / "t2_text")]

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from data import asr_path, load_jsonl, preds_path, save_jsonl
from metrics import evaluate
from prompts import build_prompt
from speech import DEFAULT_ASR, cascade_inputs

ASR_SHORT = {"whisper-large-v3-turbo": "whisperturbo"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF id, e.g. Qwen/Qwen3-VL-4B-Instruct")
ap.add_argument("--prompt-id", default="gold_style_v1")
ap.add_argument("--split", default="dev")
ap.add_argument("--n", type=int, default=200, help="subset size (seed 0); 0 = full split")
ap.add_argument("--asr", default=DEFAULT_ASR, help="ASR run name, as in asr/{split}_en_{asr}.jsonl")
ap.add_argument("--max-new-tokens", type=int, default=96)
args = ap.parse_args()

asr_file = asr_path(args.split, args.asr)
if not asr_file.exists():
    sys.exit(f"{asr_file} not found. Run: python t1_speech/run_asr.py --split {args.split}")

name = f"{ASR_SHORT.get(args.asr, args.asr)}+{args.model.split('/')[-1]}"
out = preds_path("t1_speech_cascade", args.split, name, args.prompt_id)
done = {r["id"] for r in load_jsonl(out)} if out.exists() else set()
rows = [r for r in cascade_inputs(asr_file, args.split, n=args.n or None) if r["id"] not in done]
print(f"{len(done)} done, {len(rows)} to run -> {out}")

processor = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map="auto")

for row in tqdm(rows):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(row["image_path"]).convert("RGB")},
        {"type": "text", "text": build_prompt(row["question"], args.prompt_id)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    pred = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
    save_jsonl([{"id": row["id"], "lang": row["lang"], "split": args.split,
                 "task": "t1_speech_cascade", "model": name, "prompt_id": args.prompt_id,
                 "pred": pred}], out, append=True)

res = evaluate(out, args.split)
print({k: res[k] for k in ("n", "n_missing", "bertscore_f1")})
