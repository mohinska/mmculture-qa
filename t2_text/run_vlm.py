"""Task 2 (image + text question): run a VLM on a split and write preds (no scoring here).

Kaggle/Colab:  python t2_text/run_vlm.py --model Qwen/Qwen3-VL-4B-Instruct --n 5
Qwen2.5-Omni (official baseline layout, prompt id forced to `official`):
               python t2_text/run_vlm.py --model Qwen/Qwen2.5-Omni-3B --n 0 --load-4bit
Resumable: ids already in the preds file are skipped. Needs no gold answers, so --split devtest works.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from data import load_jsonl, load_split, preds_path, save_jsonl
from prompts import OFFICIAL, PROMPTS, build_prompt

# Qwen2.5-Omni expects its own system identity (copied from the official baseline).
OMNI_SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
               "perceiving auditory and visual inputs, as well as generating text and speech.")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF id, e.g. Qwen/Qwen3-VL-4B-Instruct")
ap.add_argument("--prompt-id", default="gold_style_v1", choices=list(PROMPTS))
ap.add_argument("--split", default="dev")
ap.add_argument("--n", type=int, default=200, help="subset size (seed 0); 0 = full split")
ap.add_argument("--max-new-tokens", type=int, default=None, help="default 96 (64 for Omni, as in the official baseline)")
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--load-4bit", action="store_true", help="bitsandbytes NF4, compute dtype = --dtype")
args = ap.parse_args()

omni = "Omni" in args.model
if omni and args.prompt_id != "official":
    print(f"warning: Omni uses the official prompt layout, ignoring --prompt-id {args.prompt_id}")
    args.prompt_id = "official"
if not omni and args.prompt_id == "official":
    sys.exit("prompt id `official` is only for Qwen2.5-Omni; use plain or gold_style_v1")
max_new = args.max_new_tokens or (64 if omni else 96)

short = args.model.split("/")[-1]
out = preds_path("t2_text", args.split, short, args.prompt_id)
done = {r["id"] for r in load_jsonl(out)} if out.exists() else set()
rows = [r for r in load_split("qa", args.split, n=args.n or None) if r["id"] not in done]
print(f"{len(done)} done, {len(rows)} to run -> {out}")

# ponytail: one row per generate call, batch if the full dev run is too slow.
dtype = getattr(torch, args.dtype)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                           bnb_4bit_use_double_quant=True) if args.load_4bit else None

if omni:
    # lazy: only the Omni path needs these (transformers>=4.52, qwen-omni-utils)
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model, max_pixels=512 * 28 * 28)
    # the thinker is the text-producing half; loading it alone skips the speech decoder
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", quantization_config=quant, attn_implementation="sdpa")
    model.eval()
else:
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", quantization_config=quant)


def answer_omni(row):
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": OMNI_SYSTEM}]},
        {"role": "user", "content": [{"type": "image", "image": row["image_path"]},
                                     {"type": "text", "text": row["question"]},
                                     {"type": "text", "text": OFFICIAL}]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.inference_mode():
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=max_new)
    text_out = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
    del inputs, gen
    torch.cuda.empty_cache()          # avoids a slow creep to OOM over a long run
    return text_out


def answer_vlm(row):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(row["image_path"]).convert("RGB")},
        {"type": "text", "text": build_prompt(row["question"], args.prompt_id)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


for row in tqdm(rows):
    try:
        pred = (answer_omni if omni else answer_vlm)(row)
    except Exception as exc:                 # Omni only: one bad item must not stop the run (official script)
        if not omni:
            raise
        print(f"  [{row['id'][:12]}] failed: {type(exc).__name__}: {str(exc)[:120]}")
        pred = ""
    save_jsonl([{"id": row["id"], "answer": pred, "lang": row["lang"]}], out, append=True)

print(f"done: {out}. Score it locally with evaluate() (t2_text/notebooks/01_baselines.ipynb).")
