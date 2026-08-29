"""
run_inference.py
=======================================================================
CLI for the offline pipeline: Roman Urdu in, triage + department out.
=======================================================================

Everything runs locally. Ollama is a service on localhost, the encoder is
read from the on-disk Hugging Face cache, and the classifiers are pickles
in models_src/. No request leaves the machine.

    python run_inference.py "seena mein shadeed dard aur pasina"
    python run_inference.py "sar mein chot lagi hai" --reference "head injury"
    python run_inference.py --interactive
    python run_inference.py "pait mein dard" --json
    python run_inference.py --check          # environment only, no inference
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.offline_pipeline import (DEFAULT_THRESHOLD, MODEL_DIR, OLLAMA_MODEL,
                                  OLLAMA_URL, ollama_available, ollama_models,
                                  run)

TRIAGE_LABELS = {
    1: "EMERGENCY   (immediate)",
    2: "URGENT      (within 15 min)",
    3: "STANDARD    (within 60 min)",
    4: "NON-URGENT  (can wait / redirect)",
}


def check_environment(model: str) -> bool:
    """Report what is present before anything tries to use it."""
    print("=" * 66)
    print("OFFLINE PIPELINE  -  environment")
    print("=" * 66)
    up = ollama_available()
    print(f"  Ollama at {OLLAMA_URL:28s} {'running' if up else 'NOT REACHABLE'}")
    ok = up
    if up:
        names = ollama_models()
        print(f"  models available                     {', '.join(names) or '(none)'}")
        if model not in names:
            print(f"  [warn] {model!r} is not pulled. Either:")
            print(f"           ollama pull {model}")
            print(f"         or pass --model with one of the above.")
            ok = False
    else:
        print("  start it with:  ollama serve")

    mpath = os.path.join(MODEL_DIR, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            man = json.load(f)
        print(f"  classifiers                          {os.path.basename(MODEL_DIR)}/ "
              f"({man['estimator']}, {man['rows']} rows)")
        print(f"  encoder (from manifest)              {man['encoder']['model']} "
              f"({man['encoder']['dim']} dims)")
        for t, meta in man["targets"].items():
            print(f"    {t:14s} {len(meta['classes'])} classes")
    else:
        print(f"  classifiers                          MISSING - run: "
              f"python -m src.train")
        ok = False
    print("=" * 66)
    return ok


def render(res: dict) -> None:
    print()
    model = res.get("ollama_model")
    asked = res.get("ollama_model_requested")
    swapped = f"   (requested {asked}, not installed)" if asked and asked != model else ""
    print(f"  translator  : {model or '(none)'}{swapped}")
    print(f"  encoder     : {res.get('encoder') or '(unknown)'}")
    print(f"  input       : {res['input']}")
    tr = res.get("translation")
    print(f"  translation : {tr if tr else '(failed - see message above)'}")
    if res.get("reference"):
        print(f"  reference   : {res['reference']}")

    def block(title, pred, caveat=""):
        if not pred:
            return
        print(f"\n  {title}{caveat}")
        for target, v in pred.items():
            val = v["prediction"]
            conf = v["confidence"]
            extra = ""
            if target == "Triage_Level":
                try:
                    extra = "  " + TRIAGE_LABELS.get(int(val), "")
                except (TypeError, ValueError):
                    pass
            line = f"    {target:14s} {val}{extra}"
            if conf is not None:
                line += f"   [{conf * 100:.1f}%]"
            print(line)

    block("prediction from ROMAN URDU  (matches how the models were trained)",
          res.get("roman_urdu_prediction"))
    block("prediction from ENGLISH translation",
          res.get("english_prediction"), "  - see note below")

    sim = res.get("similarity") or {}
    if sim:
        print("\n  embedding alignment")
        if "roman_urdu_vs_english" in sim:
            v = sim["roman_urdu_vs_english"]
            passed = sim.get("passes_threshold")
            mark = "PASS" if passed else "BELOW"
            thr = sim.get("threshold", DEFAULT_THRESHOLD)
            print(f"    Roman Urdu  vs  English      {v:.4f}   "
                  f"{mark} @ {thr:.2f}")
            src = res.get("accepted_source")
            if src:
                print(f"    accepted source              "
                      f"{'ENGLISH translation' if src == 'english' else 'ROMAN URDU original'}")
        for key, label in [("english_vs_reference", "English     vs  reference  "),
                           ("roman_urdu_vs_reference", "Roman Urdu  vs  reference  ")]:
            if key in sim:
                print(f"    {label}  {sim[key]:.4f}")

    for note in res.get("notes", []):
        print(f"\n  [note] {note}")

    conf = (res.get("roman_urdu_prediction") or {}).get("Category", {}).get("confidence")
    if conf is not None and conf < 0.40:
        print("\n  [warn] department confidence under 40%. These heads were "
              "trained on 185 rows across 11 departments; treat low-confidence "
              "output as a guess. Research prototype, not a medical device.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("complaint", nargs="*", help="raw Roman Urdu complaint")
    p.add_argument("--reference", help="text to compare the translation against")
    p.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model tag")
    p.add_argument("--model-dir", default=MODEL_DIR)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--check", action="store_true",
                   help="report the environment and exit")
    args = p.parse_args()

    if args.check:
        return 0 if check_environment(args.model) else 1

    if not args.complaint and not args.interactive:
        p.error("give a complaint, or --interactive, or --check")

    if not ollama_available():
        print(f"[error] Ollama is not reachable at {OLLAMA_URL}.")
        print("        Start it with:  ollama serve")
        print("        Then re-run, or use --check to inspect the environment.")
        return 1

    if args.interactive:
        print("=" * 66)
        print("Roman Urdu complaint -> triage + department   (fully offline)")
        print(f"ollama: {args.model}   |   type 'exit' to quit")
        print("=" * 66)
        while True:
            try:
                text = input("\nComplaint: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in ("exit", "quit"):
                break
            if not text:
                continue
            render(run(text, args.reference, args.model, args.model_dir,
                       args.threshold))
        return 0

    text = " ".join(args.complaint)
    res = run(text, args.reference, args.model, args.model_dir, args.threshold)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        render(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
