"""
agent.py
--------
Q-learning agent for Senet.

Three difficulty levels correspond to different amounts of training:
    easy   →  2 000 self-play episodes
    medium → 10 000 self-play episodes
    hard   → 50 000 self-play episodes

The Q-table maps (state_tuple, action_index) → float value.
Because the full state space is huge, we use a defaultdict so unseen
states are initialised to 0.0 on demand.
"""

import random
import pickle
import os
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-learning agent.

    Hyperparameters
    ---------------
    alpha   : learning rate          (how much new info overwrites old)
    gamma   : discount factor        (how much future rewards are valued)
    epsilon : exploration rate       (probability of random action)
    epsilon_min   : floor for epsilon decay
    epsilon_decay : multiplicative decay applied after each episode
    """

    def __init__(
        self,
        player_id: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.player_id = player_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: defaultdict means unseen (s,a) pairs start at 0
        self.q_table: dict = defaultdict(float)

        # Tracking metrics per episode
        self.episode_rewards: list = []
        self.win_history: list = []     # 1 = win, 0 = loss/draw

    # ── Q-table access ───────────────────────────────────────────────────────

    def get_q(self, state: tuple, action: int) -> float:
        return self.q_table[(state, action)]

    def set_q(self, state: tuple, action: int, value: float):
        self.q_table[(state, action)] = value

    # ── Action selection ─────────────────────────────────────────────────────

    def choose_action(self, state: tuple, valid_moves: list) -> int:
        """
        Epsilon-greedy policy.
        - With probability epsilon  → choose a random valid move (explore)
        - Otherwise                 → choose the move with highest Q-value (exploit)
        Returns the board index of the piece to move.
        """
        if not valid_moves:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        # Greedy: pick action with highest Q-value
        best_action = max(valid_moves, key=lambda a: self.get_q(state, a))
        return best_action

    def choose_action_greedy(self, state: tuple, valid_moves: list) -> int:
        """Always greedy – used during evaluation / gameplay (no exploration)."""
        if not valid_moves:
            return None
        return max(valid_moves, key=lambda a: self.get_q(state, a))

    # ── Learning update ──────────────────────────────────────────────────────

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_valid_moves: list,
        done: bool,
    ):
        """
        Apply the Q-learning update rule:

            Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]

        If the episode is over there is no future state, so the target is
        simply the immediate reward.
        """
        current_q = self.get_q(state, action)

        if done or not next_valid_moves:
            target = reward
        else:
            max_next_q = max(self.get_q(next_state, a) for a in next_valid_moves)
            target = reward + self.gamma * max_next_q

        new_q = current_q + self.alpha * (target - current_q)
        self.set_q(state, action, new_q)

    # ── Epsilon decay ────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Call once per episode to reduce exploration over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, filepath: str):
        """Pickle the Q-table and metadata to disk."""
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "win_history": self.win_history,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """Restore Q-table and metadata from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data["q_table"])
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self.episode_rewards = data.get("episode_rewards", [])
        self.win_history = data.get("win_history", [])
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)

    # ── Difficulty factory ────────────────────────────────────────────────────

    @staticmethod
    def difficulty_config(level: str) -> dict:
        """
        Return training hyperparameters for each difficulty.
        More episodes + slower decay = stronger agent.
        """
        configs = {
            "easy": {
                "episodes": 2_000,
                "alpha": 0.15,
                "gamma": 0.90,
                "epsilon_decay": 0.990,
            },
            "medium": {
                "episodes": 10_000,
                "alpha": 0.10,
                "gamma": 0.95,
                "epsilon_decay": 0.995,
            },
            "hard": {
                "episodes": 50_000,
                "alpha": 0.05,
                "gamma": 0.99,
                "epsilon_decay": 0.9995,
            },
        }
        return configs.get(level, configs["medium"])
