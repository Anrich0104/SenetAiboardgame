# Senet AI — Ancient Egyptian Board Game with Q-Learning

Teaching a reinforcement-learning agent to play Senet, one of humanity's oldest board games, through pure self-play. Two Q-learning agents compete thousands of times, learning strategy from scratch with no human demonstrations or hand-coded heuristics — then you can play against the trained result in a resizable Pygame GUI.

> Originating in ancient Egypt over 5,000 years ago, Senet is a 30-square race-and-strategy game. This project frames it as a reinforcement learning problem: an agent observes the board, receives a random dice roll, and must learn which piece to move — purely from experience.

## Features

- **Q-learning self-play training** — two agents share hyperparameters and learn simultaneously, with no pre-existing strategy to imitate
- **Three difficulty levels** — Easy (2,000 episodes), Medium (10,000 episodes), and Hard (50,000 episodes), each with tuned learning rate, discount factor, and exploration decay
- **Full Senet rule implementation** — safe squares, extra-turn squares, trap squares, piece swapping, and win detection
- **Pygame GUI** with three modes:
  - Human vs AI
  - AI vs AI (watch two trained agents play)
  - Interactive "How to Play" tutorial
- **Freely resizable window** — the board and UI scale proportionally
- **Training visualizations** — win rate and cumulative reward graphs generated automatically per difficulty level
- **Pre-trained models included** so you can start playing immediately without training first

## How It Works

The project follows Mitchell's (1997) Task/Experience/Performance framework:

| Component | Description |
|---|---|
| **Task (T)** | Play Senet — observe the board, accept a dice roll (1–5), advance a piece, handle special squares, and win by moving all five pieces past square 30 |
| **Experience (E)** | Self-play only — two Q-learning agents compete over thousands of full games, with no human data or heuristics |
| **Performance (P)** | Win rate, average cumulative reward, and Q-table coverage, tracked over rolling 200-episode windows |

The agent uses tabular **Q-learning** (Watkins, 1989), updating estimated state-action values via the Bellman equation. The board state is a 30-integer tuple (`{-1, 0, +1}` per square), used directly as a hashable Q-table key. Exploration starts at ε = 1.0 (fully random) and decays each episode toward a floor, shifting the agent from exploration to exploiting what it has learned.

**Reward structure:** Win `+1.00` · Lose `−1.00` · Valid move `+0.05` · Invalid move `−0.10` · Per-step `−0.01`

### Training configuration

| Difficulty | Episodes | α (learning rate) | γ (discount) |
|---|---|---|---|
| Easy | 2,000 | 0.15 | 0.90 |
| Medium | 10,000 | 0.10 | 0.95 |
| Hard | 50,000 | 0.05 | 0.99 |

The Hard agent accumulates on the order of millions of unique Q-table entries and shows clear strategic behavior — advancing pieces toward the exit, swapping opponent pieces off the board, and avoiding the trap square.

## Project Structure

```
SenetBoardGame/
├── main.py            # Entry point — trains missing models, then launches the GUI
├── game_engine.py      # Core rules: board state, legal moves, special squares, win detection
├── agent.py             # Q-learning agent: action selection, learning updates, save/load
├── training.py          # Self-play training loop + training graph generation
├── gui.py                # Pygame GUI: rendering, input handling, game modes, tutorial
├── utils.py              # Shared helpers: smoothing, logging, progress bars
├── models/               # Pre-trained Q-tables (agent_easy.pkl, agent_medium.pkl, agent_hard.pkl)
├── graphs/               # Training graphs generated per difficulty (created on training)
└── Senet_Code_Guide.pdf  # Plain-language walkthrough of how the files fit together
```

## Getting Started

### Prerequisites

- Python 3.9+
- [pygame](https://www.pygame.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

```bash
pip install pygame numpy matplotlib
```

### Run the game

Pre-trained models are already included in `models/`, so you can launch straight into the GUI:

```bash
python main.py --skip-train
```

Or simply run `python main.py` — it will automatically train any missing difficulty models (Easy → Medium → Hard) before launching.

### Training only

To train models without opening the GUI:

```bash
python main.py --train-only            # trains all three difficulty levels
python main.py --train-only --level hard   # trains a single level
```

Or run the training script directly:

```bash
python training.py --level easy
python training.py --level medium
python training.py --level hard
python training.py --level all
```

Trained Q-tables are saved to `models/agent_<level>.pkl`, per-episode logs to `logs/log_<level>.csv`, and win-rate/reward graphs to `graphs/training_<level>.png`.

## How to Play

- Click a piece (highlighted with a gold outline) to select it — its legal destination squares light up in green.
- Click a highlighted destination square to move the piece there.
- **Square 15** is Safe — the opponent cannot swap you off it.
- **Square 20** grants an Extra Turn.
- **Square 27** is a Trap — landing here sends your piece back to the start.
- A piece is removed from the board once it moves past square 30.

Use the in-game menu to choose Human vs AI, AI vs AI, difficulty level, or open the How to Play tutorial.

## Results

Across self-play training, win rate stabilizes near 50% for symmetric agents sharing hyperparameters — this reflects learned parity between two improving opponents rather than a failure to learn. Early episodes (high ε) show near-random play and negative average reward due to the per-step penalty; as ε decays, agents increasingly exploit learned Q-values, and average reward rises as pieces move validly toward the exit.

## Limitations & Future Work

- **State space explosion** — tabular Q-learning can't generalize to unseen board states; a Deep Q-Network (DQN) could address this
- **Dice stochasticity** — the same board position can call for different actions depending on the roll, and the current state representation doesn't encode the roll itself
- **Co-evolving opponents** — since both agents improve simultaneously, there's no fixed learning target
- **Simplified ruleset** — historical Senet rules aren't fully documented, so some rules here are interpretive
- Planned directions: DQN-based generalization, incorporating the dice roll into the state representation, and curriculum training against progressively stronger fixed opponents

## References

Mitchell, T. (1997). *Machine Learning*. Watkins, C. (1989). *Learning from Delayed Rewards*. Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*. Piccione, R. (1980). *In Search of the Meaning of Senet*. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*.

