"""Task 1 (image + spoken question): transcribe the audio with Whisper and write asr/.

Kaggle:  python t1_speech/run_asr.py --split dev
Names prompt (own file):  python t1_speech/run_asr.py --initial-prompt "Petra, Souq Waqif" --tag whisper-large-v3-turbo+names
Resumable: ids already in the asr file are skipped.
"""
import argparse

from speech import DEFAULT_ASR, asr_summary, default_backend, run_asr, wer_refs

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_ASR, help="key of ASR_MODELS in speech.py")
ap.add_argument("--split", default="dev")
ap.add_argument("--n", type=int, default=0, help="subset size; 0 = full split")
ap.add_argument("--seed", type=int, default=0, help="subset seed (0 = same ids as the T2_Text runs)")
ap.add_argument("--backend", default=None, choices=["mlx", "ct2", "hf"], help="default: auto")
ap.add_argument("--initial-prompt", default=None, help="context text for the decoder (mlx / ct2 only)")
ap.add_argument("--tag", default=None, help="output name instead of --model; required with --initial-prompt")
ap.add_argument("--batch-size", type=int, default=16)
args = ap.parse_args()

print("backend:", args.backend or default_backend())
path = run_asr(args.split, model=args.model, n=args.n or None, seed=args.seed,
               backend=args.backend, initial_prompt=args.initial_prompt, tag=args.tag,
               batch_size=args.batch_size)
if wer_refs(args.split):    # WER is our own analysis; splits without text questions get none
    print(asr_summary(path, args.split))
