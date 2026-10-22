"""Task 2 (image + text question): run a VLM on a dev subset and write preds.

Kaggle:  python t2_text/run_vlm.py --model Qwen/Qwen3-VL-4B-Instruct --n 5
Resumable: ids already in the preds file are skipped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from data import load_jsonl, load_split, preds_path, save_jsonl
from metrics import evaluate
from prompts import build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF id, e.g. Qwen/Qwen3-VL-4B-Instruct")
ap.add_argument("--prompt-id", default="gold_style_v1")
ap.add_argument("--split", default="dev")
ap.add_argument("--n", type=int, default=200, help="subset size (seed 0); 0 = full split")
ap.add_argument("--max-new-tokens", type=int, default=96)
args = ap.parse_args()

short = args.model.split("/")[-1]
out = preds_path("t2_text", args.split, short, args.prompt_id)
done = {r["id"] for r in load_jsonl(out)} if out.exists() else set()
rows = [r for r in load_split("qa", args.split, n=args.n or None) if r["id"] not in done]
print(f"{len(done)} done, {len(rows)} to run -> {out}")

# fp16: T4 has no bf16. ponytail: one row per generate call, batch if the full dev run is too slow.
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
                 "task": "t2_text", "model": short, "prompt_id": args.prompt_id,
                 "pred": pred}], out, append=True)

res = evaluate(out, args.split)
print({k: res[k] for k in ("n", "n_missing", "bertscore_f1")})
