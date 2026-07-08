"""
main.py
-------
Entry point for the Senet AI project.

Workflow
--------
1. Check which difficulty models are already trained (models/ directory).
2. Train any missing models (easy → medium → hard).
3. Launch the Pygame GUI.

Usage
-----
    python main.py              ← train missing models + launch GUI
    python main.py --skip-train ← skip training, just launch GUI
    python main.py --train-only ← train all levels, no GUI
"""

import os
import sys
import argparse

# Ensure local imports resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def models_missing() -> list:
    """Return list of difficulty levels whose model files are absent."""
    missing = []
    for level in ("easy", "medium", "hard"):
        path = os.path.join(MODEL_DIR, f"agent_{level}.pkl")
        if not os.path.exists(path):
            missing.append(level)
    return missing


def run_training(levels: list):
    """Train the specified difficulty levels."""
    from training import train
    for level in levels:
        print(f"\n[main.py] Training '{level}' agent …")
        train(level, verbose=True)


def launch_gui():
    """Launch the Pygame GUI."""
    from gui import SenetGUI
    gui = SenetGUI()
    gui.run()


def main():
    parser = argparse.ArgumentParser(description="Senet AI — Train & Play")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and launch GUI immediately")
    parser.add_argument("--train-only", action="store_true",
                        help="Train all difficulty levels, then exit (no GUI)")
    parser.add_argument("--level", choices=["easy", "medium", "hard", "all"],
                        default=None,
                        help="Train a specific level only")
    args = parser.parse_args()

    print("=" * 60)
    print("  SENET — Ancient Egyptian Board Game with Q-Learning AI")
    print("=" * 60)

    # ── Training phase ────────────────────────────────────────────────────────
    if args.train_only:
        levels = ["easy", "medium", "hard"] if args.level in (None, "all") \
                 else [args.level]
        run_training(levels)
        print("\n[main.py] All training complete. Exiting.")
        return

    if not args.skip_train:
        missing = models_missing()
        if args.level and args.level != "all":
            # User specified a level: retrain only that one
            run_training([args.level])
        elif missing:
            print(f"\n[main.py] Missing models: {missing}")
            print("[main.py] Starting training … (this may take a few minutes)\n")
            run_training(missing)
        else:
            print("\n[main.py] All models already trained. Launching GUI …")
    else:
        print("\n[main.py] Skipping training (--skip-train flag set).")

    # ── GUI phase ─────────────────────────────────────────────────────────────
    print("\n[main.py] Launching Senet GUI …")
    launch_gui()


if __name__ == "__main__":
    main()
