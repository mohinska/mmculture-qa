# MMCultureQA (SemEval-2027)

UCU "Artificial Intelligence" course project.

Task: answer culture-specific questions about an image. The question comes as text
(Task 2, Textual VQA) or as speech (Task 1, Spoken VQA). Answers are short and open-ended.

- Track: English. The English questions released so far are about MENA countries, hence `mena` in data paths.
- Metric: BERTScore F1. For Spoken VQA we also report ASR WER.
- Task page: https://mmcultureqa-semeval27.github.io/
- Data: https://huggingface.co/datasets/QCRI/MMCQA-SemEval27 (CC BY-NC, not stored in this repo)

## Approach

| | Input | Pipeline |
|---|---|---|
| Task 1, Spoken VQA, cascade | image + audio question | Whisper ASR → transcript → the Textual VQA model |
| Task 1, Spoken VQA, end-to-end | image + audio question | audio-capable multimodal model, no ASR |
| Task 2, Textual VQA | image + text question | VLM, zero-shot → prompting → fine-tuning |

## Structure

```
common.py         shared helpers: paths, load_split(), checks, jsonl I/O, wer(), bertscore(), evaluate()
download_data.py  downloads the English data from Hugging Face into data/
notebooks/        00_* shared, 10_* Task 1 Spoken VQA, 20_* Task 2 Textual VQA
results/          metrics (.json) for every run, committed
requirements.txt
.env.example      template for paths and HF_TOKEN
```

Not tracked by git (see `.gitignore`):

```
data/     dataset              (under MMCQA_ROOT, default: repo folder)
asr/      ASR transcripts      (under MMCQA_OUT, default: same as MMCQA_ROOT)
preds/    model answers        (under MMCQA_OUT)
```

## Setup

```bash
git clone https://github.com/mohinska/mmculture-qa.git
cd mmculture-qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env`:

| Variable | Meaning |
|---|---|
| `HF_TOKEN` | Hugging Face read token (https://huggingface.co/settings/tokens) |
| `MMCQA_ROOT` | folder with `data/`. Empty = repo folder |
| `MMCQA_OUT` | where `asr/` and `preds/` are written. Empty = same as `MMCQA_ROOT` |

## Data

```bash
python download_data.py dev                 # text + speech (default --task all)
python download_data.py dev --task qa       # Task 2 Textual VQA: qa jsonl + images
python download_data.py dev --task sqa      # Task 1 Spoken VQA: sqa jsonl + images + audio (+ qa jsonl for WER)
python download_data.py dev devtest train   # several splits
python download_data.py --list              # list files in the HF repo
```

Splits: train 10k, dev 1k, devtest 1k. Audio in train and dev is synthetic (TTS);
the test set will have real recordings and no `answer` / `country` / `category` fields.

Check that data is in place:

```python
from common import *
rows = load_split("sqa", "dev")
print(len(rows), check_ids(rows), check_media(rows))
```

## File formats

- Transcripts: `asr/{split}_en_{model}.jsonl`, fields `id, lang, split, model, transcript, wer`
- Predictions: `preds/{task}_{split}_en_{model}_{prompt_id}.jsonl`, fields `id, lang, split, task, model, prompt_id, pred`,
  where `task` is `t1_speech_cascade`, `t1_speech_e2e` or `t2_text`
- Metrics: `results/<same name>.json` with `bertscore_f1/p/r`, `n`, `by_category`, `date`

Use `asr_path()` and `preds_path()` from `common.py` to build these names.

## Rules

- Never edit `data/` or someone else's files in `asr/` and `preds/`.
- New experiment = new file. Nothing gets overwritten.
- Quick runs: `load_split(..., n=200, seed=0)`, the same subset for both tasks.
- Changing `common.py`: pull first, separate commit, tell the other person.
- Notebook outputs and comments in English. Clear large outputs before committing.

## Kaggle (GPU runs)

Add `HF_TOKEN` in Add-ons → Secrets, then in the notebook:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

```bash
!git clone https://github.com/mohinska/mmculture-qa.git
%cd mmculture-qa
!pip install -q -r requirements.txt
%env MMCQA_OUT=/kaggle/working
!python download_data.py dev --task sqa
```

Download `preds/` from the notebook output afterwards.

## Team

| | Name | GitHub | Role |
|:-:|---|---|---|
| <img src="https://github.com/mohinska.png?size=80" width="48" alt="mohinska"/> | Maryna Ohinska | [@mohinska](https://github.com/mohinska) | Task 2, Textual VQA: VLM, prompting, fine-tuning |
| <img src="https://github.com/PechenenkoYara.png?size=80" width="48" alt="PechenenkoYara"/> | Yaryna Pechenenko | [@PechenenkoYara](https://github.com/PechenenkoYara) | Task 1, Spoken VQA: ASR, cascade, end-to-end |
