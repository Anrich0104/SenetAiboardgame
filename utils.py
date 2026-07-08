"""
utils.py
--------
Shared utility functions: smoothing, file I/O, progress display.
"""

import os
import csv
import numpy as np


def moving_average(data: list, window: int) -> list:
    """Compute a centred moving average with the given window size."""
    if len(data) < window:
        window = max(1, len(data))
    arr = np.array(data, dtype=float)
    kernel = np.ones(window) / window
    # Use 'same' mode so output length matches input length
    smoothed = np.convolve(arr, kernel, mode="same")

    # Fix boundary distortion caused by zero-padding
    half = window // 2
    for i in range(half):
        smoothed[i] = np.mean(arr[:i + half + 1])
    for i in range(len(arr) - half, len(arr)):
        smoothed[i] = np.mean(arr[i - half:])

    return smoothed.tolist()


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_training_log(win_log: list, reward_log: list, filepath: str):
    """Write per-episode win/reward data to a CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "win", "reward"])
        for i, (w, r) in enumerate(zip(win_log, reward_log), start=1):
            writer.writerow([i, w, f"{r:.4f}"])


def print_progress_bar(current: int, total: int, length: int = 30, suffix: str = ""):
    """Print an ASCII progress bar to stdout."""
    filled = int(length * current / total)
    bar = "█" * filled + "░" * (length - filled)
    pct = 100 * current / total
    print(f"\r  [{bar}] {pct:5.1f}%  ep {current:,}/{total:,}  {suffix}", end="", flush=True)
    if current == total:
        print()   # Newline at completion


def board_to_features(board: tuple) -> np.ndarray:
    """
    Convert a board state tuple into a numpy feature vector.
    Used if a neural network extension is added later.
    """
    return np.array(board, dtype=np.float32)


def count_pieces(board: tuple, player: int) -> int:
    """Count how many pieces `player` has on the board."""
    return sum(1 for v in board if v == player)


def compute_win_rate(win_history: list, last_n: int = 200) -> float:
    """Return the win rate over the last N episodes."""
    if not win_history:
        return 0.0
    subset = win_history[-last_n:]
    return sum(subset) / len(subset)
