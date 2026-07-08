"""
gui.py
------
Pygame-based GUI for Senet — Ancient Egyptian Board Game with Q-Learning AI.

Supports three game modes:
    • Human vs AI  (human is always Player 1 / Blue)
    • AI vs AI     (watch two trained agents play against each other)
    • How to Play  (interactive tutorial guide with step-by-step explanations)

Difficulty levels:
    • Easy   — AI trained for 2,000 episodes (beginner-friendly)
    • Medium — AI trained for 10,000 episodes (balanced challenge)
    • Hard   — AI trained for 50,000 episodes (very strong opponent)

Special squares on the board:
    • Square 15 (index 14) — SAFE: opponent cannot swap you off this square
    • Square 20 (index 19) — EXTRA TURN: landing here grants an extra roll
    • Square 27 (index 26) — TRAP: sends your piece all the way back to start

Movement (two-step):
    1. Click a piece with a gold outline  → it is selected; its reachable
       destination squares are highlighted in bright green.
    2. Click one of those green destination squares → the piece moves there.
    Clicking a different own piece in step 2 re-selects it instead.

Window:
    The window is freely resizable — drag any edge or corner.
    All board and UI elements scale proportionally to the current size.

Controls
--------
    Click highlighted pieces then destination squares to move.
    Buttons: New Game | Human vs AI | AI vs AI | How to Play | Easy | Medium | Hard
"""

import sys
import os
import random

# ── Guard: graceful error if pygame is not installed ────────────────────────
try:
    import pygame
except ImportError:
    print("\n[ERROR] pygame is not installed.")
    print("  Install it with:  pip install pygame")
    print("  Then re-run:      python main.py\n")
    sys.exit(1)

# Ensure local modules (game_engine, agent) can be imported from the same folder
sys.path.insert(0, os.path.dirname(__file__))

from game_engine import (
    SenetGame, PLAYER1, PLAYER2,
    BOARD_SIZE, SAFE_SQUARE, EXTRA_TURN_SQ, TRAP_SQUARE
)
from agent import QLearningAgent

# ── Colour palette ───────────────────────────────────────────────────────────
SAND            = (210, 180, 140)   # Default board cell background
DARK_WOOD       = ( 80,  50,  20)   # Board cell border
GOLD            = (218, 165,  32)   # Active button / selectable-piece highlight
DEST_HIGHLIGHT  = ( 50, 205,  50)   # Bright green — valid destination squares
BLUE_PIECE      = ( 30, 100, 200)   # Player 1 piece colour
RED_PIECE       = (200,  40,  40)   # Player 2 piece colour
WHITE           = (255, 255, 255)
BLACK           = (  0,   0,   0)
LIGHT_GRAY      = (220, 220, 220)
DARK_GRAY       = ( 80,  80,  80)
SAFE_COL        = (144, 238, 144)   # Light green  — safe square background
TRAP_COL        = (255, 160, 122)   # Light salmon — trap square background
EXTRA_COL       = (173, 216, 230)   # Light blue   — extra-turn square background

# Tutorial overlay colours
TIP_BG     = (240, 230, 180)   # Warm parchment for tip cards
TIP_BORDER = (139, 100,  20)   # Dark gold border for tip cards

# ── Base layout (used to calculate scale factor) ─────────────────────────────
# These are the "design" dimensions the layout was originally built for.
# At runtime everything is multiplied by a scale factor derived from the
# actual window size so the UI fits any resolution.
BASE_CELL    = 70     # Design cell size in pixels
BASE_MARGIN  = 20     # Design outer margin in pixels
BASE_BTN_H   = 40     # Design button height in pixels
BASE_INFO_H  = 160    # Design info-panel height in pixels
COLS         = 10     # Board columns (fixed by the game rules)
ROWS         = 3      # Board rows    (fixed by the game rules)

# Starting window size — user can drag to resize freely
INIT_W = COLS * BASE_CELL + 2 * BASE_MARGIN           # 740
INIT_H = ROWS * BASE_CELL + 2 * BASE_MARGIN + BASE_INFO_H  # 430

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── How To Play content ──────────────────────────────────────────────────────
# Each dict is one page of the tutorial.
# "highlight" is None or one of "safe" | "extra" | "trap" — when set, that
# special square gets an extra thick border on the live board behind the card.
HOW_TO_PLAY_PAGES = [
    {
        "title": "Welcome to Senet! 🏺",
        "lines": [
            "Senet is one of the oldest known board",
            "games — played in ancient Egypt over",
            "5,000 years ago!",
            "",
            "You are the BLUE player (Player 1).",
            "Your goal: move ALL 5 of your pieces",
            "off the far end of the board before",
            "the RED AI opponent does the same.",
            "",
            "Press  ▶ Next  to learn how to play.",
        ],
        "highlight": None,
    },
    {
        "title": "The Board 📋",
        "lines": [
            "The board has 30 squares arranged in",
            "3 rows of 10, travelled in an S-shape:",
            "",
            "  Row 1: squares  1-10  (left → right)",
            "  Row 2: squares 11-20  (right → left)",
            "  Row 3: squares 21-30  (left → right)",
            "",
            "Pieces start interleaved at squares",
            "1-10: Blue on odd, Red on even slots.",
            "First to clear all pieces wins! 🏆",
        ],
        "highlight": None,
    },
    {
        "title": "Rolling the Dice 🎲",
        "lines": [
            "Ancient Senet used 4 throwing sticks",
            "(each blank on one side, marked other).",
            "",
            "This game simulates that with a random",
            "roll of 1 – 5 each turn.",
            "",
            "The number you roll is how many",
            "squares your chosen piece will move.",
            "",
            "The dice roll is shown in the info",
            "panel below the board after each turn.",
        ],
        "highlight": None,
    },
    {
        "title": "Moving Your Pieces 🕹️",
        "lines": [
            "On YOUR turn:",
            "",
            "  Step 1 — Click a piece with a GOLD",
            "  outline to select it.",
            "",
            "  Step 2 — Green squares appear showing",
            "  where it can move. Click one to move.",
            "",
            "  Click a different gold piece at any",
            "  time to re-select instead.",
            "",
            "You CANNOT move onto your OWN pieces.",
        ],
        "highlight": None,
    },
    {
        "title": "Safe Square 🛡️  (Square 15)",
        "lines": [
            "The GREEN square (square 15) is SAFE.",
            "",
            "If your piece is sitting on this",
            "square, the opponent CANNOT swap it",
            "off — you are completely protected.",
            "",
            "Try to use it as a shelter when the",
            "AI is closing in on your pieces!",
            "",
            "Tip: The AI knows about this square",
            "too, so expect it to use it as well.",
        ],
        "highlight": "safe",
    },
    {
        "title": "Extra Turn Square ⭐  (Square 20)",
        "lines": [
            "The BLUE square (square 20) grants",
            "an EXTRA TURN to whoever lands on it.",
            "",
            "Landing here means you get to roll",
            "and move again immediately — great",
            "for racing pieces to the end!",
            "",
            "The status bar will show:",
            "  '★ Extra turn!'",
            "when this bonus is triggered.",
        ],
        "highlight": "extra",
    },
    {
        "title": "Trap Square ☠️  (Square 27)",
        "lines": [
            "The RED/SALMON square (square 27) is",
            "a TRAP — landing here sends that",
            "piece all the way back to square 1!",
            "",
            "This can cost you several turns, so",
            "plan your moves carefully near the",
            "end of the board.",
            "",
            "Good news: the AI can also land on",
            "the trap — watch for it!",
        ],
        "highlight": "trap",
    },
    {
        "title": "Winning the Game 🏆",
        "lines": [
            "A piece exits the board by moving",
            "past square 30 (index 29).",
            "",
            "The first player to remove ALL 5",
            "of their pieces wins the game.",
            "",
            "After winning, click  New Game",
            "to play again, or try a harder",
            "difficulty level to test yourself!",
            "",
            "Good luck — pharaoh is watching! 👁️",
        ],
        "highlight": None,
    },
    {
        "title": "AI Difficulty Levels 🤖",
        "lines": [
            "Three AI opponents are available:",
            "",
            "  🟢 Easy   — trained 2,000 rounds",
            "     Makes mistakes; good for learning",
            "",
            "  🟡 Medium — trained 10,000 rounds",
            "     Solid play; a fair challenge",
            "",
            "  🔴 Hard   — trained 50,000 rounds",
            "     Plans ahead; very tough to beat",
            "",
            "Start on Easy and work your way up!",
        ],
        "highlight": None,
    },
    {
        "title": "AI vs AI Mode 🤖🤖",
        "lines": [
            "Select  AI vs AI  to watch two",
            "trained agents battle each other.",
            "",
            "This is a great way to see advanced",
            "Senet strategy in action before you",
            "play yourself.",
            "",
            "You can change the difficulty at any",
            "time — both AIs will use that level.",
            "",
            "Press  ✔ Got it!  to close this guide",
            "and start playing. Have fun! 🎉",
        ],
        "highlight": None,
    },
]


# ── Scaling helpers ───────────────────────────────────────────────────────────

def compute_scale(win_w: int, win_h: int) -> float:
    """
    Compute a uniform scale factor so the board fits the current window.

    We scale based on whichever axis is the tighter constraint (width or
    height) to ensure nothing gets clipped.  A minimum of 0.4 prevents the
    window from becoming unusably small.
    """
    # How much of the height the board+UI needs at base scale
    board_h_base = ROWS * BASE_CELL + 2 * BASE_MARGIN + BASE_INFO_H
    board_w_base = COLS * BASE_CELL + 2 * BASE_MARGIN

    scale_x = win_w / board_w_base
    scale_y = win_h / board_h_base
    return max(0.4, min(scale_x, scale_y))


def cell_rect(pos: int, scale: float) -> pygame.Rect:
    """
    Return a pygame.Rect for board cell at linear index `pos`, scaled.

    The board follows an S-shaped path:
      • Row 0 (pos  0-9):  left → right
      • Row 1 (pos 10-19): right → left  (col reversed)
      • Row 2 (pos 20-29): left → right
    """
    cell  = int(BASE_CELL   * scale)
    margin = int(BASE_MARGIN * scale)

    row = pos // COLS
    col = pos % COLS
    if row == 1:
        col = COLS - 1 - col   # Mirror middle row for S-path

    x = margin + col * cell
    y = margin + row * cell
    return pygame.Rect(x, y, cell, cell)


class SenetGUI:
    """
    Main GUI controller for Senet.

    Responsibilities:
      • Initialise a resizable Pygame window, fonts, and buttons
      • Load pre-trained Q-learning agents from disk
      • Handle user input with two-step move selection:
            click piece (gold) → destinations turn green → click destination
      • Drive the AI turn logic with a short visual delay
      • Recompute layout on every frame so resizing works seamlessly
      • Render board, pieces, info panel, buttons, and How-to-Play overlay
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Senet-Ancient Egyptian Board Game")

        # RESIZABLE flag lets the user drag window edges freely
        self.screen = pygame.display.set_mode((INIT_W, INIT_H), pygame.RESIZABLE)
        self.clock  = pygame.time.Clock()

        # ── Game state ────────────────────────────────────────────────────────
        self.game        = SenetGame()
        self.mode        = "human_vs_ai"   # 'human_vs_ai' | 'ai_vs_ai'
        self.difficulty  = "medium"
        self.selected    = None    # Board index of the currently selected piece
        self.move_sources = []     # Pieces the current player CAN move (gold outline)
        self.destinations = []     # Where the selected piece CAN go  (green highlight)
        self.status_msg  = "Welcome! Click 'How to Play' to learn the rules."
        self.dice_result = 0
        self.game_over   = False
        self.agents      = {}      # difficulty → QLearningAgent

        # ── Tutorial state ────────────────────────────────────────────────────
        self.how_to_play_active = False
        self.htp_page           = 0

        self._load_agents()
        self._new_game()

    # =========================================================================
    # Helpers — layout derived from current window size
    # =========================================================================

    def _scale(self) -> float:
        """Return the current uniform scale factor based on window dimensions."""
        w, h = self.screen.get_size()
        return compute_scale(w, h)

    def _dims(self):
        """
        Return a dict of all layout measurements scaled to the current window.

        Centralising this avoids scattering scale arithmetic across every
        drawing method — each draw call just calls _dims() once.
        """
        s      = self._scale()
        cell   = int(BASE_CELL   * s)
        margin = int(BASE_MARGIN * s)
        btn_h  = int(BASE_BTN_H  * s)
        info_h = int(BASE_INFO_H * s)

        board_w = COLS * cell + 2 * margin
        board_h = ROWS * cell + 2 * margin

        # Font sizes scale with the window, clamped to sensible limits
        fs_sm = max(10, int(14 * s))
        fs_md = max(12, int(18 * s))
        fs_lg = max(16, int(26 * s))

        return {
            "s": s, "cell": cell, "margin": margin,
            "btn_h": btn_h, "info_h": info_h,
            "board_w": board_w, "board_h": board_h,
            "fs_sm": fs_sm, "fs_md": fs_md, "fs_lg": fs_lg,
            "win_w": self.screen.get_width(),
            "win_h": self.screen.get_height(),
        }

    def _fonts(self, d: dict):
        """
        Return (font_sm, font_md, font_lg) sized for the current scale.

        Fonts are created fresh when needed; pygame font creation is fast
        so this has no perceptible cost.
        """
        return (
            pygame.font.SysFont("Arial", d["fs_sm"]),
            pygame.font.SysFont("Arial", d["fs_md"], bold=True),
            pygame.font.SysFont("Arial", d["fs_lg"], bold=True),
        )

    def _button_rects(self, d: dict) -> list:
        """
        Compute scaled button rects dynamically from current dimensions.

        Buttons are distributed evenly across the window width so they never
        overflow or leave awkward gaps after resizing.
        """
        board_bottom = d["board_h"] + 10   # Y just below the board
        n   = 7        # Total number of buttons
        gap = max(4, int(6 * d["s"]))
        total_gap = gap * (n - 1)
        bw  = max(60, (d["win_w"] - 2 * d["margin"] - total_gap) // n)
        x   = d["margin"]

        labels_actions_colours = [
            ("New Game",    "new_game",    ( 60, 100,  60)),
            ("Human vs AI", "human_vs_ai", ( 30,  80, 160)),
            ("AI vs AI",    "ai_vs_ai",    (120,  40, 120)),
            ("How to Play", "how_to_play", ( 80,  60, 130)),
            ("Easy",        "easy",        ( 60, 130,  60)),
            ("Medium",      "medium",      (200, 140,   0)),
            ("Hard",        "hard",        (160,  30,  30)),
        ]

        btns = []
        for label, action, colour in labels_actions_colours:
            r = pygame.Rect(x, board_bottom, bw, d["btn_h"])
            btns.append({"rect": r, "label": label, "action": action, "colour": colour})
            x += bw + gap
        return btns

    # =========================================================================
    # Model loading
    # =========================================================================

    def _load_agents(self):
        """
        Load each pre-trained Q-table from models/ if the file exists.
        Epsilon is set to 0.0 so the AI plays greedily (no exploration).
        Missing files are silently skipped; a random fallback is used at play.
        """
        for level in ("easy", "medium", "hard"):
            path = os.path.join(MODEL_DIR, f"agent_{level}.pkl")
            if os.path.exists(path):
                agent = QLearningAgent(PLAYER2)
                agent.load(path)
                agent.epsilon = 0.0
                self.agents[level] = agent

    # =========================================================================
    # Game management
    # =========================================================================

    def _new_game(self):
        """Reset all game state and prepare the first turn."""
        self.game.reset()
        self.selected     = None
        self.move_sources = []
        self.destinations = []
        self.game_over    = False
        self.dice_result  = 0
        self._roll_and_prepare()

    def _roll_and_prepare(self):
        """
        Roll the dice for the current player, compute which pieces they can
        move (move_sources), and update the status message.
        If no moves exist the turn is automatically passed after a short delay.
        """
        self.dice_result    = self.game.roll_dice()
        self.game.dice_roll = self.dice_result
        self.move_sources   = self.game.get_valid_moves()   # Piece indices player can move
        self.destinations   = []   # Reset destinations — no piece selected yet
        self.selected       = None

        player_label = (
            "Player 1 (Blue)" if self.game.current_player == PLAYER1
            else "Player 2 (Red)"
        )
        self.status_msg = f"{player_label}'s turn — Roll: {self.dice_result}"

        if not self.move_sources:
            self.status_msg += "  (No valid moves — passing)"
            self._pass_turn()

    def _pass_turn(self):
        """Switch player and re-roll when there are genuinely no valid moves."""
        self.game.current_player = -self.game.current_player
        pygame.time.delay(600)
        self._roll_and_prepare()

    def _compute_destinations(self, from_pos: int) -> list:
        """
        Return the list of destination squares reachable by the piece at
        `from_pos` given the current dice roll.

        A destination is the single square `from_pos + dice` (or off-board).
        We return it as a list so the drawing code can treat it uniformly.
        The game engine already validated `from_pos` is in move_sources, so
        the destination is guaranteed legal.
        """
        dest = from_pos + self.dice_result
        if dest >= BOARD_SIZE:
            # Piece would exit the board — represent as a virtual square 30
            # so the player can click anywhere off the right edge to confirm.
            # In practice we handle this by showing the piece as "exit-ready"
            # and auto-moving on selection (see _handle_cell_click).
            return []   # No square to click — auto-move handled below
        return [dest]

    def _apply_move(self, from_pos: int):
        """
        Execute the move from `from_pos`, check for game-over / extra-turn,
        then hand off to the next turn.
        """
        _, reward, done, info = self.game.move_piece(from_pos)
        self.selected     = None
        self.destinations = []
        self.move_sources = []

        if done:
            winner = (
                "Player 1 (Blue)" if self.game.winner == PLAYER1
                else "Player 2 (Red)"
            )
            self.status_msg = f"🏆 {winner} wins!"
            self.game_over = True
        elif info.get("extra_turn"):
            self.status_msg += "  ★ Extra turn!"
            self._roll_and_prepare()
        else:
            self._roll_and_prepare()

    # =========================================================================
    # AI turn
    # =========================================================================

    def _ai_move(self):
        """
        Let the Q-learning agent (or random fallback) pick and execute a move.
        A 500 ms delay makes the AI move visually perceptible to the human.
        """
        agent = self.agents.get(self.difficulty)
        state = self.game.get_state()
        moves = self.game.get_valid_moves()

        if not moves:
            return

        action = (
            agent.choose_action_greedy(state, moves)
            if agent else random.choice(moves)
        )

        pygame.time.delay(500)
        self._apply_move(action)

    # =========================================================================
    # Main loop
    # =========================================================================

    def run(self):
        """
        Main game loop at 30 FPS.

        Each frame:
          1. Cap frame rate
          2. Process pygame events
          3. Redraw everything (layout recalculated each frame for resize)
          4. Trigger AI move if it is not the human's turn
        """
        while True:
            self.clock.tick(30)
            self._handle_events()
            self._draw()

            if not self.game_over and not self.how_to_play_active:
                is_human_turn = (
                    self.mode == "human_vs_ai"
                    and self.game.current_player == PLAYER1
                )
                if not is_human_turn:
                    self._ai_move()

    # =========================================================================
    # Event handling
    # =========================================================================

    def _handle_events(self):
        """Process all pending pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # VIDEORESIZE is fired while dragging; RESIZABLE mode handles the
            # surface automatically — we just need to keep using get_size().
            if event.type == pygame.VIDEORESIZE:
                # Re-create the surface at the new size (required for older pygame)
                self.screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE
                )

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)

    def _handle_click(self, pos: tuple):
        """
        Route a left-click to the appropriate handler.

        Priority:
          1. How-to-play overlay (if open)
          2. Control buttons
          3. Board cells (human turn only)
        """
        if self.how_to_play_active:
            self._handle_htp_click(pos)
            return

        d    = self._dims()
        btns = self._button_rects(d)

        for btn in btns:
            if btn["rect"].collidepoint(pos):
                self._handle_button(btn["action"])
                return

        if self.game_over:
            return
        if self.mode == "human_vs_ai" and self.game.current_player != PLAYER1:
            return

        s = d["s"]
        for cell_idx in range(BOARD_SIZE):
            if cell_rect(cell_idx, s).collidepoint(pos):
                self._handle_cell_click(cell_idx)
                return

    def _handle_button(self, action: str):
        """Respond to a control-button action string."""
        if action == "new_game":
            self._new_game()
        elif action == "human_vs_ai":
            self.mode = "human_vs_ai"
            self.status_msg = "Mode: Human (Blue) vs AI (Red)"
            self._new_game()
        elif action == "ai_vs_ai":
            self.mode = "ai_vs_ai"
            self.status_msg = "Mode: AI vs AI — watch and learn!"
            self._new_game()
        elif action == "how_to_play":
            self.how_to_play_active = True
            self.htp_page = 0
        elif action in ("easy", "medium", "hard"):
            self.difficulty = action
            if action not in self.agents:
                self.status_msg = (
                    f"⚠ No '{action}' model found. "
                    f"Run: python training.py --level {action}"
                )
            else:
                self.status_msg = f"Difficulty set to {action.capitalize()}"

    def _handle_cell_click(self, cell_idx: int):
        """
        Two-step move interaction for the human player:

        Phase A — no piece selected yet:
          • Click on a gold-outlined piece (in move_sources) → select it,
            compute and show destination squares in green.
          • Click anywhere else → ignored.

        Phase B — a piece is already selected:
          • Click on a green destination square → execute the move.
          • Click on a different gold-outlined piece → re-select it.
          • Click on the selected piece again → deselect.
          • Click anywhere else → deselect.

        Special case — piece exits the board:
          When dice would carry a piece past square 29, there is no
          destination square to click.  In that case we auto-move
          immediately when the piece is selected.
        """
        board  = self.game.board
        player = self.game.current_player

        if self.selected is None:
            # ── Phase A: select a piece ───────────────────────────────────────
            if cell_idx in self.move_sources:
                self.selected    = cell_idx
                dests = self._compute_destinations(cell_idx)
                if not dests:
                    # Piece exits board — no destination square, auto-move now
                    self._apply_move(cell_idx)
                else:
                    self.destinations = dests
                    self.status_msg = (
                        f"Piece at sq.{cell_idx+1} selected — "
                        f"click a green square to move"
                    )
        else:
            # ── Phase B: piece already selected ──────────────────────────────
            if cell_idx in self.destinations:
                # Valid destination clicked — execute move
                self._apply_move(self.selected)

            elif cell_idx in self.move_sources and cell_idx != self.selected:
                # Re-select a different own piece
                self.selected    = cell_idx
                dests = self._compute_destinations(cell_idx)
                if not dests:
                    self._apply_move(cell_idx)
                else:
                    self.destinations = dests
                    self.status_msg = (
                        f"Piece at sq.{cell_idx+1} selected — "
                        f"click a green square to move"
                    )

            else:
                # Anything else → cancel selection
                self.selected     = None
                self.destinations = []
                player_label = (
                    "Player 1 (Blue)" if player == PLAYER1 else "Player 2 (Red)"
                )
                self.status_msg = (
                    f"{player_label}'s turn — Roll: {self.dice_result}  "
                    f"(click a gold piece)"
                )

    # =========================================================================
    # How-to-Play click handler
    # =========================================================================

    def _handle_htp_click(self, pos: tuple):
        """Handle ◀ Back, ▶ Next, and ✔ Got it! clicks inside the overlay."""
        d      = self._dims()
        win_w  = d["win_w"]
        win_h  = d["win_h"]
        margin = d["margin"]

        card_x = margin + 30
        card_y = margin + 20
        card_w = win_w - 2 * (margin + 30)
        card_h = win_h - 2 * (margin + 20)

        nav_y  = card_y + card_h - 55
        btn_w  = max(90, int(110 * d["s"]))
        btn_h  = max(28, int(36  * d["s"]))
        gap    = 12

        back_rect = pygame.Rect(card_x + gap, nav_y, btn_w, btn_h)
        next_rect = pygame.Rect(card_x + card_w - btn_w - gap, nav_y, btn_w, btn_h)

        if back_rect.collidepoint(pos) and self.htp_page > 0:
            self.htp_page -= 1
        elif next_rect.collidepoint(pos):
            if self.htp_page < len(HOW_TO_PLAY_PAGES) - 1:
                self.htp_page += 1
            else:
                self.how_to_play_active = False   # Close on last page

    # =========================================================================
    # Drawing — main dispatch
    # =========================================================================

    def _draw(self):
        """
        Full redraw every frame.  Layout is recomputed from the current window
        size each call so resizing is always correct.
        """
        d              = self._dims()
        font_sm, font_md, font_lg = self._fonts(d)

        self.screen.fill((245, 230, 200))   # Warm parchment background

        self._draw_board(d, font_sm)
        self._draw_pieces(d, font_md)
        self._draw_info_panel(d, font_sm, font_md)
        self._draw_buttons(d, font_sm)

        if self.how_to_play_active:
            self._draw_htp_overlay(d, font_sm, font_md, font_lg)

        pygame.display.flip()

    # =========================================================================
    # Drawing — board
    # =========================================================================

    def _draw_board(self, d: dict, font_sm):
        """
        Draw all 30 board cells.

        Highlights:
          • Gold border  — piece the human can pick up (move_sources)
          • Green border — square the selected piece can move TO (destinations)
          • Thick yellow — HTP tutorial spotlight on a special square
          • Normal wood border — everything else
        """
        s = d["s"]

        # Which special square (if any) should be spotlit for the HTP page
        htp_idx = None
        if self.how_to_play_active:
            h = HOW_TO_PLAY_PAGES[self.htp_page].get("highlight")
            if h == "safe":
                htp_idx = SAFE_SQUARE
            elif h == "extra":
                htp_idx = EXTRA_TURN_SQ
            elif h == "trap":
                htp_idx = TRAP_SQUARE

        for idx in range(BOARD_SIZE):
            r = cell_rect(idx, s)

            # Background colour based on square type
            if idx == SAFE_SQUARE:
                bg = SAFE_COL
            elif idx == TRAP_SQUARE:
                bg = TRAP_COL
            elif idx == EXTRA_TURN_SQ:
                bg = EXTRA_COL
            else:
                bg = SAND

            pygame.draw.rect(self.screen, bg, r)

            # Border priority: HTP spotlight > destination > movable source > normal
            if idx == htp_idx:
                pygame.draw.rect(self.screen, (255, 255, 0), r, max(3, int(5*s)))
            elif idx in self.destinations:
                pygame.draw.rect(self.screen, DEST_HIGHLIGHT, r, max(3, int(4*s)))
            elif idx in self.move_sources and self.selected is None:
                # Gold border on all movable pieces before one is selected
                pygame.draw.rect(self.screen, GOLD, r, max(2, int(4*s)))
            elif idx == self.selected:
                # Selected piece keeps its gold border
                pygame.draw.rect(self.screen, GOLD, r, max(2, int(4*s)))
            else:
                pygame.draw.rect(self.screen, DARK_WOOD, r, 1)

            # Square number (1-indexed) — top-left of cell
            num = font_sm.render(str(idx + 1), True, DARK_GRAY)
            self.screen.blit(num, (r.x + 3, r.y + 3))

            # Special-square text label — centred toward the bottom of the cell
            label_offset_y = r.centery + int(12 * s)
            if idx == SAFE_SQUARE:
                lbl = font_sm.render("SAFE", True, (0, 100, 0))
                self.screen.blit(lbl, (r.centerx - lbl.get_width()//2, label_offset_y))
            elif idx == TRAP_SQUARE:
                lbl = font_sm.render("TRAP", True, (139, 0, 0))
                self.screen.blit(lbl, (r.centerx - lbl.get_width()//2, label_offset_y))
            elif idx == EXTRA_TURN_SQ:
                lbl = font_sm.render("+TURN", True, (0, 0, 139))
                self.screen.blit(lbl, (r.centerx - lbl.get_width()//2, label_offset_y))

    # =========================================================================
    # Drawing — pieces
    # =========================================================================

    def _draw_pieces(self, d: dict, font_md):
        """
        Draw each piece as a filled circle with a player-number label.

        Radius scales with cell size.  Selected piece gets a gold ring;
        all others get a white ring.
        """
        s      = d["s"]
        radius = max(8, int(20 * s))

        for idx in range(BOARD_SIZE):
            val = self.game.board[idx]
            if val == 0:
                continue

            r  = cell_rect(idx, s)
            cx = r.centerx
            cy = r.centery - int(8 * s)   # Shift up slightly for the square label

            colour = BLUE_PIECE if val == PLAYER1 else RED_PIECE
            border = GOLD if idx == self.selected else WHITE

            pygame.draw.circle(self.screen, colour, (cx, cy), radius)
            pygame.draw.circle(self.screen, border, (cx, cy), radius, max(2, int(3*s)))

            sym      = "1" if val == PLAYER1 else "2"
            sym_surf = font_md.render(sym, True, WHITE)
            self.screen.blit(
                sym_surf,
                (cx - sym_surf.get_width()  // 2,
                 cy - sym_surf.get_height() // 2)
            )

    # =========================================================================
    # Drawing — info panel
    # =========================================================================

    def _draw_info_panel(self, d: dict, font_sm, font_md):
        """
        Draw the status/legend panel below the button row.

        Contains:
          • Status message (whose turn, dice, win announcement)
          • Mode / difficulty / dice summary
          • Colour legend for the three special squares
        """
        s        = d["s"]
        panel_y  = d["board_h"] + d["btn_h"] + int(20 * s)
        margin   = d["margin"]
        mode_lbl = "Human vs AI" if self.mode == "human_vs_ai" else "AI vs AI"
        diff_lbl = self.difficulty.capitalize()

        # Instruction hint when it's the human's turn and no piece is selected
        if (not self.game_over
                and self.mode == "human_vs_ai"
                and self.game.current_player == PLAYER1):
            if self.selected is None:
                hint = "  ← click a gold piece to select"
            else:
                hint = "  ← click a green square to move"
            display_msg = self.status_msg + hint
        else:
            display_msg = self.status_msg

        status_surf = font_md.render(display_msg, True, BLACK)
        self.screen.blit(status_surf, (margin, panel_y + int(50 * s)))

        info = (
            f"Mode: {mode_lbl}   |   "
            f"Difficulty: {diff_lbl}   |   "
            f"Dice: {self.dice_result}"
        )
        info_surf = font_sm.render(info, True, DARK_GRAY)
        self.screen.blit(info_surf, (margin, panel_y + int(76 * s)))

        # Colour legend
        sq  = max(10, int(16 * s))   # Colour swatch size
        gap = max(16, int(22 * s))   # Vertical gap between legend rows
        legend_y = panel_y + int(100 * s)
        for colour, label in [
            (SAFE_COL,  "Safe square (sq. 15)"),
            (TRAP_COL,  "Trap — back to start (sq. 27)"),
            (EXTRA_COL, "Extra turn (sq. 20)"),
        ]:
            pygame.draw.rect(self.screen, colour, (margin, legend_y, sq, sq))
            lbl = font_sm.render(label, True, DARK_GRAY)
            self.screen.blit(lbl, (margin + sq + 6, legend_y))
            legend_y += gap

    # =========================================================================
    # Drawing — control buttons
    # =========================================================================

    def _draw_buttons(self, d: dict, font_sm):
        """
        Draw the row of control buttons, highlighting the active mode and
        active difficulty in gold, and the How to Play button when the
        guide is open.
        """
        for btn in self._button_rects(d):
            colour = btn["colour"]
            if btn["action"] == self.mode or btn["action"] == self.difficulty:
                colour = GOLD
            if btn["action"] == "how_to_play" and self.how_to_play_active:
                colour = GOLD

            pygame.draw.rect(self.screen, colour, btn["rect"], border_radius=6)
            pygame.draw.rect(self.screen, WHITE,  btn["rect"], 2, border_radius=6)

            lbl = font_sm.render(btn["label"], True, WHITE)
            self.screen.blit(
                lbl,
                (btn["rect"].centerx - lbl.get_width()  // 2,
                 btn["rect"].centery - lbl.get_height() // 2)
            )

    # =========================================================================
    # Drawing — How-to-Play overlay
    # =========================================================================

    def _draw_htp_overlay(self, d: dict, font_sm, font_md, font_lg):
        """
        Draw the tutorial guide as a full-window semi-transparent overlay.

        Layout:
          • Dark translucent backdrop over the whole window
          • Rounded parchment card filling most of the window
          • Page title + divider
          • Body text
          • Page counter centred at bottom of card
          • ◀ Back and ▶ Next / ✔ Got it! navigation buttons
        """
        win_w  = d["win_w"]
        win_h  = d["win_h"]
        margin = d["margin"]
        s      = d["s"]

        # Semi-transparent backdrop
        overlay = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
        overlay.fill((20, 20, 50, 200))
        self.screen.blit(overlay, (0, 0))

        # Card geometry — leave a comfortable border around the window edges
        pad    = margin + 30
        card_x = pad
        card_y = pad
        card_w = win_w - 2 * pad
        card_h = win_h - 2 * pad

        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)
        pygame.draw.rect(self.screen, TIP_BG,     card_rect, border_radius=12)
        pygame.draw.rect(self.screen, TIP_BORDER, card_rect, 3, border_radius=12)

        page = HOW_TO_PLAY_PAGES[self.htp_page]

        # Title
        title_surf = font_lg.render(page["title"], True, (80, 40, 0))
        self.screen.blit(
            title_surf,
            (card_x + card_w // 2 - title_surf.get_width() // 2,
             card_y + int(18 * s))
        )

        # Divider under title
        divider_y = card_y + int(58 * s)
        pygame.draw.line(
            self.screen, TIP_BORDER,
            (card_x + 20, divider_y),
            (card_x + card_w - 20, divider_y), 2
        )

        # Body text lines
        line_h = max(18, int(22 * s))
        text_y = divider_y + int(12 * s)
        for line in page["lines"]:
            if line == "":
                text_y += int(8 * s)
                continue
            line_surf = font_sm.render(line, True, (50, 30, 10))
            self.screen.blit(line_surf, (card_x + int(24 * s), text_y))
            text_y += line_h

        # Navigation buttons
        nav_y = card_y + card_h - int(55 * s)
        btn_w = max(90, int(110 * s))
        btn_h = max(28, int(36  * s))
        gap   = 12

        # Page counter centred between the two nav buttons
        page_txt  = f"{self.htp_page + 1} / {len(HOW_TO_PLAY_PAGES)}"
        pg_surf   = font_sm.render(page_txt, True, DARK_GRAY)
        self.screen.blit(
            pg_surf,
            (card_x + card_w // 2 - pg_surf.get_width() // 2,
             nav_y + (btn_h - pg_surf.get_height()) // 2)
        )

        # ◀ Back (greyed out on first page)
        back_rect   = pygame.Rect(card_x + gap, nav_y, btn_w, btn_h)
        back_colour = (120, 100, 60) if self.htp_page > 0 else (180, 170, 150)
        pygame.draw.rect(self.screen, back_colour, back_rect, border_radius=8)
        pygame.draw.rect(self.screen, TIP_BORDER,  back_rect, 2, border_radius=8)
        back_lbl = font_sm.render("◀ Back", True, WHITE)
        self.screen.blit(
            back_lbl,
            (back_rect.centerx - back_lbl.get_width()  // 2,
             back_rect.centery - back_lbl.get_height() // 2)
        )

        # ▶ Next  OR  ✔ Got it!
        next_rect    = pygame.Rect(card_x + card_w - btn_w - gap, nav_y, btn_w, btn_h)
        is_last      = self.htp_page == len(HOW_TO_PLAY_PAGES) - 1
        next_label   = "✔ Got it!" if is_last else "▶ Next"
        next_colour  = (60, 130, 60) if is_last else (80, 60, 130)
        pygame.draw.rect(self.screen, next_colour, next_rect, border_radius=8)
        pygame.draw.rect(self.screen, TIP_BORDER,  next_rect, 2, border_radius=8)
        next_lbl = font_sm.render(next_label, True, WHITE)
        self.screen.blit(
            next_lbl,
            (next_rect.centerx - next_lbl.get_width()  // 2,
             next_rect.centery - next_lbl.get_height() // 2)
        )
