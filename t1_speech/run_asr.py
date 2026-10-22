"""Task 1 (image + spoken question): transcribe the audio with Whisper and write asr/.

Kaggle:  python t1_speech/run_asr.py --split dev
Resumable: ids already in the asr file are skipped.
"""
import argparse

from speech import DEFAULT_ASR, asr_summary, default_backend, run_asr

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_ASR, help="key of ASR_MODELS in speech.py")
ap.add_argument("--split", default="dev")
ap.add_argument("--n", type=int, default=0, help="subset size (seed 0); 0 = full split")
ap.add_argument("--backend", default=None, help="mlx | ct2 | hf (default: auto)")
ap.add_argument("--batch-size", type=int, default=16)
args = ap.parse_args()

print("backend:", args.backend or default_backend())
path = run_asr(args.split, model=args.model, n=args.n or None,
               backend=args.backend, batch_size=args.batch_size)
if args.split != "test":
    print(asr_summary(path, args.split))
