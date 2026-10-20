"""Download the English data of MMCQA-SemEval27 from Hugging Face into $MMCQA_ROOT/data.

Usage:
    python download_data.py                          # dev, text + speech
    python download_data.py dev devtest --task qa    # text VQA only: qa jsonl + images
    python download_data.py dev --task sqa           # spoken VQA: sqa jsonl + audio + images
    python download_data.py --list                   # print repo files, download nothing

--task sqa also fetches the qa jsonl: its `question` is the reference for WER.

Auth: the dataset may be gated. Put HF_TOKEN=... in .env or run `huggingface-cli login`.
"""
import argparse
import fnmatch
import os
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from common import DATA

REPO_ID = "QCRI/MMCQA-SemEval27"


def patterns(split: str, task: str) -> list[str]:
    qa = [f"qa/mena/{split}_en.jsonl", f"*/qa/mena/{split}_en.jsonl"]
    sqa = [f"sqa/mena/{split}_en.jsonl", f"*/sqa/mena/{split}_en.jsonl"]
    images = [f"*images_mena_{split}*.zip"]
    audio = [f"*audio_mena_en_{split}*.zip"]
    if task == "qa":
        return qa + images
    if task == "sqa":
        return sqa + qa + images + audio  # qa jsonl = WER references
    return qa + sqa + images + audio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splits", nargs="*", default=["dev"], help="train / dev / devtest (default: dev)")
    ap.add_argument("--task", choices=["qa", "sqa", "all"], default="all",
                    help="qa = text VQA, sqa = spoken VQA, all = both (default)")
    ap.add_argument("--list", action="store_true", help="only print files in the HF repo")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    files = list_repo_files(REPO_ID, repo_type="dataset", token=token)

    if args.list:
        print("\n".join(files))
        return

    wanted = sorted({f for s in args.splits for p in patterns(s, args.task)
                     for f in files if fnmatch.fnmatch(f, p)})
    if not wanted:
        sys.exit(f"Nothing matched for splits={args.splits}, task={args.task}. "
                 "Run with --list and fix patterns().")

    DATA.mkdir(parents=True, exist_ok=True)
    for f in wanted:
        print("download", f)
        path = Path(hf_hub_download(REPO_ID, f, repo_type="dataset", token=token, local_dir=DATA))
        if path.suffix == ".zip":
            print("  unzip ->", path.parent)
            with zipfile.ZipFile(path) as z:
                z.extractall(path.parent)
    print("done:", DATA)


if __name__ == "__main__":
    main()
