"""Turn a preds file into an upload-ready Codabench submission and run the official format checker.

Usage:
    python shared/make_submission.py --preds preds/t2_text_devtest_en_<model>_<prompt>.jsonl [--split devtest]

Writes submissions/<preds stem>/prediction.jsonl and prediction.zip (only prediction.jsonl inside).
n=200 subsets fail the id check by design: the checker needs every id of the split (preds must cover the full split: run_vlm.py --n 0).
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from data import LANG, REPO, SPLITS, load_jsonl, require_answers, save_jsonl, split_path

CHECKER = REPO / "shared" / "format_checker" / "check_format.py"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--split", default="devtest", choices=SPLITS)
    args = ap.parse_args()

    src = load_jsonl(args.preds)
    require_answers(src, args.preds)
    rows = [{"id": r["id"], "answer": "" if r.get("answer") is None else str(r["answer"]), "lang": LANG}
            for r in src]
    out = REPO / "submissions" / args.preds.stem
    jsonl, zip_path = out / "prediction.jsonl", out / "prediction.zip"
    save_jsonl(rows, jsonl)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl, "prediction.jsonl")

    res = subprocess.run([sys.executable, str(CHECKER), "--pred", str(zip_path), "--lang", LANG,
                          "--ids", str(split_path("qa", args.split))], capture_output=True, text=True)
    print(res.stdout + res.stderr, end="")
    if res.returncode:
        sys.exit(res.returncode)
    print(f"\nReady: {zip_path}\nUpload it on Codabench: Dev-test phase, Language field = {LANG}.")


if __name__ == "__main__":
    main()
