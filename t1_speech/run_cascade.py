"""Task 1 cascade: ASR transcript + image -> the T2_Text VLM -> preds (no scoring here).

Same model loading, prompt and generation as answer_vlm() in t2_text/run_vlm.py;
only the question comes from asr/ instead of the text file.

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
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from data import asr_path, load_jsonl, preds_path, save_jsonl
from prompts import PROMPTS, build_prompt
from speech import ASR_SHORT, DEFAULT_ASR, cascade_inputs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF id, e.g. Qwen/Qwen3-VL-4B-Instruct")
ap.add_argument("--prompt-id", default="gold_style_v1", choices=[p for p in PROMPTS if p != "official"])
ap.add_argument("--split", default="dev")
ap.add_argument("--n", type=int, default=200, help="subset size (seed 0); 0 = full split")
ap.add_argument("--asr", default=DEFAULT_ASR, help="ASR run name, as in asr/{split}_en_{asr}.jsonl")
ap.add_argument("--max-new-tokens", type=int, default=96)
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes NF4, compute dtype = --dtype")
args = ap.parse_args()

if "Omni" in args.model:
    sys.exit("Qwen2.5-Omni takes the audio itself: that is the end-to-end run, not the cascade")
asr_file = asr_path(args.split, args.asr)
if not asr_file.exists():
    sys.exit(f"{asr_file} not found. Run: python t1_speech/run_asr.py --split {args.split}")

name = f"{ASR_SHORT.get(args.asr, args.asr)}+{args.model.split('/')[-1]}"
out = preds_path("t1_speech_cascade", args.split, name, args.prompt_id)
done = {r["id"] for r in load_jsonl(out)} if out.exists() else set()
rows = [r for r in cascade_inputs(asr_file, args.split, n=args.n or None) if r["id"] not in done]
print(f"{len(done)} done, {len(rows)} to run -> {out}")

dtype = getattr(torch, args.dtype)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                           bnb_4bit_use_double_quant=True) if args.load_4bit else None
processor = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model, torch_dtype=dtype, device_map="auto", quantization_config=quant)

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
    save_jsonl([{"id": row["id"], "answer": pred, "lang": row["lang"]}], out, append=True)

print(f"done: {out}. Score it with evaluate() (t1_speech/notebooks/11_cascade.ipynb).")
