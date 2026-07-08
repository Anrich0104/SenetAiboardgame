"""
training.py
-----------
Self-play training loop for the Senet Q-learning agent.

Run directly:
    python training.py --level easy
    python training.py --level medium
    python training.py --level hard
    python training.py --level all    ← trains all three and saves each

Trained Q-tables are saved to:
    models/agent_easy.pkl
    models/agent_medium.pkl
    models/agent_hard.pkl

Training graphs are saved to:
    graphs/training_<level>.png
"""

import os
import sys
import argparse
import random

# Resolve imports when run from project root
sys.path.insert(0, os.path.dirname(__file__))

from game_engine import SenetGame, PLAYER1, PLAYER2
from agent import QLearningAgent
from utils import (
    moving_average,
    ensure_dir,
    save_training_log,
    print_progress_bar,
)

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend – safe on all platforms
import matplotlib.pyplot as plt
import numpy as np


# ── Directories ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
GRAPH_DIR  = os.path.join(os.path.dirname(__file__), "graphs")
LOG_DIR    = os.path.join(os.path.dirname(__file__), "logs")


def train(level: str = "medium", verbose: bool = True) -> QLearningAgent:
    """
    Train a Q-learning agent via self-play at the specified difficulty level.

    Two agents share the same Q-table and play against each other.
    After every episode, epsilon is decayed for both.

    Returns the trained agent (player 1 perspective).
    """
    ensure_dir(MODEL_DIR)
    ensure_dir(GRAPH_DIR)
    ensure_dir(LOG_DIR)

    cfg = QLearningAgent.difficulty_config(level)
    episodes     = cfg["episodes"]
    alpha        = cfg["alpha"]
    gamma        = cfg["gamma"]
    epsilon_decay = cfg["epsilon_decay"]

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Training Senet Agent — Difficulty: {level.upper()}")
        print(f"  Episodes: {episodes:,}  |  α={alpha}  |  γ={gamma}")
        print(f"{'='*55}")

    # Both agents share hyperparameters; agent2 mirrors agent1 but plays P2
    agent1 = QLearningAgent(PLAYER1,  alpha=alpha, gamma=gamma,
                            epsilon=1.0, epsilon_decay=epsilon_decay)
    agent2 = QLearningAgent(PLAYER2,  alpha=alpha, gamma=gamma,
                            epsilon=1.0, epsilon_decay=epsilon_decay)

    win_log    = []   # 1 = P1 won, 0 = P1 lost
    reward_log = []   # Cumulative reward for P1 per episode

    MAX_STEPS = 500   # Guard against infinite games

    for ep in range(1, episodes + 1):
        game = SenetGame()
        state = game.reset()
        ep_reward = 0.0
        steps = 0

        while not game.done and steps < MAX_STEPS:
            steps += 1

            # Roll dice
            roll = game.roll_dice()
            game.dice_roll = roll

            current_player = game.current_player
            agent = agent1 if current_player == PLAYER1 else agent2

            valid_moves = game.get_valid_moves()

            if not valid_moves:
                # No legal moves – pass turn
                game.current_player = -current_player
                continue

            action = agent.choose_action(state, valid_moves)
            next_state, reward, done, info = game.move_piece(action)

            # Compute next valid moves for the Q-update target
            if not done:
                next_roll = game.roll_dice()
                game.dice_roll = next_roll
                next_valid = game.get_valid_moves()
            else:
                next_valid = []

            # Update Q-table
            agent.update(state, action, reward, next_state, next_valid, done)

            # From P1's perspective: if P2 got a reward that means P1 lost
            if current_player == PLAYER1:
                ep_reward += reward
            else:
                ep_reward -= reward   # Mirror for logging

            state = next_state

        # Episode wrap-up
        winner = game.winner
        win_log.append(1 if winner == PLAYER1 else 0)
        reward_log.append(ep_reward)

        agent1.decay_epsilon()
        agent2.decay_epsilon()

        # Progress display
        if verbose and (ep % max(1, episodes // 20) == 0 or ep == episodes):
            win_rate = sum(win_log[-200:]) / min(len(win_log), 200)
            avg_r    = np.mean(reward_log[-200:])
            print_progress_bar(ep, episodes, suffix=
                f"WinRate(last200)={win_rate:.2%}  AvgReward={avg_r:.3f}  "
                f"ε={agent1.epsilon:.4f}")

    # Attach logs to agent1 for saving
    agent1.win_history     = win_log
    agent1.episode_rewards = reward_log

    # Save model
    model_path = os.path.join(MODEL_DIR, f"agent_{level}.pkl")
    agent1.save(model_path)
    if verbose:
        print(f"\n  ✓ Model saved → {model_path}")
        print(f"  ✓ Q-table entries: {len(agent1.q_table):,}")

    # Save log
    log_path = os.path.join(LOG_DIR, f"log_{level}.csv")
    save_training_log(win_log, reward_log, log_path)

    # Generate graphs
    graph_path = _plot_training(win_log, reward_log, level, episodes)
    if verbose:
        print(f"  ✓ Graphs saved → {graph_path}\n")

    return agent1


def _plot_training(win_log, reward_log, level, episodes):
    """Save a two-panel training graph: win rate and average reward vs episodes."""
    window = min(500, max(50, episodes // 20))

    smoothed_wins    = moving_average(win_log, window)
    smoothed_rewards = moving_average(reward_log, window)
    x = range(len(smoothed_wins))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"Senet Q-Learning Training — {level.capitalize()} Difficulty\n"
        f"({episodes:,} episodes, smoothing window={window})",
        fontsize=13, fontweight="bold"
    )

    # Win-rate panel
    ax1.plot(x, smoothed_wins, color="#2196F3", linewidth=1.5, label="Win Rate (P1)")
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50% baseline")
    ax1.set_ylabel("Win Rate", fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(x, smoothed_wins, 0.5,
                     where=[v > 0.5 for v in smoothed_wins],
                     alpha=0.15, color="#2196F3")

    # Reward panel
    ax2.plot(x, smoothed_rewards, color="#4CAF50", linewidth=1.5, label="Avg Reward (P1)")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Cumulative Reward", fontsize=11)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, f"training_{level}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def train_all_levels(verbose: bool = True):
    """Convenience wrapper: train easy, medium, and hard sequentially."""
    for level in ("easy", "medium", "hard"):
        train(level, verbose=verbose)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Senet Q-learning agents.")
    parser.add_argument(
        "--level", default="medium",
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty level to train (default: medium)"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    if args.level == "all":
        train_all_levels(verbose=not args.quiet)
    else:
        train(args.level, verbose=not args.quiet)
