"""
game_engine.py
--------------
Core Senet game logic: board state, rules, movement, and win detection.
Senet is a 30-square Egyptian board game played on a 3x10 grid in an S-shape path.
"""

import random

# ── Board constants ──────────────────────────────────────────────────────────
BOARD_SIZE = 30
NUM_PIECES = 5

# Special squares (0-indexed)
SAFE_SQUARE   = 14   # Square 15 – landing here is safe (cannot be swapped off)
EXTRA_TURN_SQ = 19   # Square 20 – grants an extra turn
TRAP_SQUARE   = 26   # Square 27 – sends the piece back to square 0

# Cell values
EMPTY    =  0
PLAYER1  =  1   # Human / agent 1
PLAYER2  = -1   # Opponent / agent 2

class SenetGame:
    """
    Manages one game of Senet.

    Board is a flat list of 30 integers:
        0  → empty
        1  → Player 1 piece
       -1  → Player 2 piece

    Pieces move along the S-shaped path:
        Row 0 (squares  0-9):  left → right
        Row 1 (squares 10-19): right → left
        Row 2 (squares 20-29): left → right
    A piece exits the board once it passes square 29.
    """

    def __init__(self):
        self.reset()

    # ── Initialisation ───────────────────────────────────────────────────────

    def reset(self):
        """Reset board to starting position and return initial state."""
        self.board = [EMPTY] * BOARD_SIZE
        self.current_player = PLAYER1
        self.extra_turn = False
        self.done = False
        self.winner = None
        self.dice_roll = 0

        # Players alternate pieces: [0,2,4,6,8] for P1, [1,3,5,7,9] for P2
        for i in range(NUM_PIECES):
            self.board[i * 2]     = PLAYER1
            self.board[i * 2 + 1] = PLAYER2

        return self.get_state()

    # ── State representation ─────────────────────────────────────────────────

    def get_state(self):
        """Return the board as a tuple (hashable for Q-table keys)."""
        return tuple(self.board)

    # ── Dice ─────────────────────────────────────────────────────────────────

    @staticmethod
    def roll_dice():
        """Simulate throwing 4 binary sticks → 1–5 (ancient Senet mechanic)."""
        return random.randint(1, 5)

    # ── Movement helpers ─────────────────────────────────────────────────────

    def get_valid_moves(self, player=None, roll=None):
        """
        Return a list of board indices where the current player has a piece
        that can legally move by `roll` squares.

        A move is legal when the destination is:
            • off the board (piece exits – allowed), OR
            • empty, OR
            • occupied by ONE opponent piece (swap), AND
            • NOT a safe square occupied by the opponent.
        Blocked if 2+ consecutive opponent pieces are in the path? → not implemented
        here for simplicity; blocking is handled at the destination only.
        """
        if player is None:
            player = self.current_player
        if roll is None:
            roll = self.dice_roll

        valid = []
        for pos in range(BOARD_SIZE):
            if self.board[pos] != player:
                continue
            dest = pos + roll
            if dest >= BOARD_SIZE:
                valid.append(pos)          # Piece exits – always legal
                continue
            dest_val = self.board[dest]
            if dest_val == EMPTY:
                valid.append(pos)
            elif dest_val == -player:
                # Cannot swap if destination is the safe square
                if dest != SAFE_SQUARE:
                    valid.append(pos)
            # dest_val == player → own piece → blocked, not valid
        return valid

    def move_piece(self, from_pos, roll=None):
        """
        Execute a move from `from_pos` by `roll` squares.
        Returns (new_state, reward, done, info_dict).
        """
        if roll is None:
            roll = self.dice_roll

        player = self.current_player
        valid_moves = self.get_valid_moves(player, roll)

        if from_pos not in valid_moves:
            # Invalid move penalty
            return self.get_state(), -0.1, self.done, {"invalid": True}

        dest = from_pos + roll

        # Remove piece from source
        self.board[from_pos] = EMPTY
        reward = 0.05  # Small reward for a valid step

        if dest >= BOARD_SIZE:
            # Piece exits the board – just removed
            pass
        else:
            dest_val = self.board[dest]
            if dest_val == -player:
                # Swap: opponent piece goes back to from_pos
                self.board[from_pos] = -player

            # Apply special square effects
            if dest == TRAP_SQUARE:
                self.board[0] = player   # Sent back to start
                # (do not place at dest)
            else:
                self.board[dest] = player

        # Check win condition
        p1_count = self.board.count(PLAYER1)
        p2_count = self.board.count(PLAYER2)

        if p1_count == 0:
            self.done = True
            self.winner = PLAYER1
            reward = 1.0 if player == PLAYER1 else -1.0
        elif p2_count == 0:
            self.done = True
            self.winner = PLAYER2
            reward = 1.0 if player == PLAYER2 else -1.0
        else:
            reward -= 0.01  # Small step penalty to encourage efficiency

        # Determine next player
        if not self.done:
            if dest == EXTRA_TURN_SQ and dest < BOARD_SIZE:
                self.extra_turn = True        # Same player goes again
            else:
                self.extra_turn = False
                self.current_player = -player  # Switch

        return self.get_state(), reward, self.done, {
            "invalid": False,
            "extra_turn": self.extra_turn,
            "winner": self.winner,
        }

    # ── Utility ──────────────────────────────────────────────────────────────

    def clone(self):
        """Return a deep copy of this game (used during lookahead / training)."""
        g = SenetGame.__new__(SenetGame)
        g.board = self.board[:]
        g.current_player = self.current_player
        g.extra_turn = self.extra_turn
        g.done = self.done
        g.winner = self.winner
        g.dice_roll = self.dice_roll
        return g

    def __repr__(self):
        rows = []
        for r in range(3):
            row = self.board[r * 10: r * 10 + 10]
            if r == 1:
                row = row[::-1]   # S-path reversal for display
            rows.append(" ".join(f"{v:+d}" for v in row))
        return "\n".join(rows)
