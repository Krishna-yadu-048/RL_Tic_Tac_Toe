import sys
import os
import customtkinter as ctk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import TicTacToeEnv
from src.agent import QLearningAgent, RandomAgent


class TicTacToeGUI:
    def __init__(self):
        # ── window setup ──
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Tic-Tac-Toe  ·  Q-Learning Agent")
        self.root.resizable(False, False)

        # ── colors ──
        self.COLOR_BG = "#1a1a2e"
        self.COLOR_CARD = "#16213e"
        self.COLOR_X = "#e94560"
        self.COLOR_O = "#0f3460"
        self.COLOR_O_TEXT = "#53a8e2"
        self.COLOR_GRID = "#2a2a4a"
        self.COLOR_HOVER = "#2a2a4a"
        self.COLOR_WIN = "#4ecca3"
        self.COLOR_TEXT = "#eaeaea"
        self.COLOR_MUTED = "#8a8a9a"
        self.COLOR_LEARN = "#f0a500"

        self.root.configure(fg_color=self.COLOR_BG)

        # ── game state ──
        self.env = TicTacToeEnv()
        self.agent = QLearningAgent(learning_rate=0.1, discount_factor=0.95, epsilon=0.1)
        self.agent_loaded = False
        self.buttons = []
        self.game_active = False

        # score tracking
        self.score = {"wins": 0, "losses": 0, "draws": 0}

        # difficulty: "easy" = random agent, "hard" = Q-learning agent
        self.difficulty = "hard"

        # learning mode: when ON, agent updates Q-table after each game
        self.learning_enabled = True

        # move history for the current game — needed for Q-table updates
        # each entry: {"player": 1 or -1, "state": tuple, "action": int}
        self.move_history = []

        # path to save updated Q-table
        self.model_path = self._find_model_path()

        # try loading the trained model
        self._load_agent()

        # ── build the UI ──
        self._build_ui()
        self._new_game()

    def _find_model_path(self):
        """Figure out where to save/load the Q-table."""
        candidates = [
            "models/q_table.pkl",
            os.path.join(os.path.dirname(__file__), "..", "models", "q_table.pkl"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return os.path.abspath(path)
        # default: create in project root's models folder
        default = os.path.join(os.path.dirname(__file__), "..", "models", "q_table.pkl")
        return os.path.abspath(default)

    def _load_agent(self):
        """Try to load the Q-table from the models directory."""
        if os.path.exists(self.model_path):
            self.agent.load(self.model_path)
            self.agent_loaded = True
        else:
            print("⚠ No trained model found. Run 'python src/train.py' first.")
            print("  The agent will start from scratch and learn as you play.")
            self.agent_loaded = True  # still usable, just empty Q-table

    def _build_ui(self):
        """Construct all the UI elements."""

        # ── main container ──
        self.main_frame = ctk.CTkFrame(self.root, fg_color=self.COLOR_BG)
        self.main_frame.pack(padx=30, pady=20)

        # ── title ──
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="TIC-TAC-TOE",
            font=ctk.CTkFont(family="Helvetica", size=28, weight="bold"),
            text_color=self.COLOR_TEXT,
        )
        title_label.pack(pady=(0, 2))

        subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="vs Q-Learning Agent",
            font=ctk.CTkFont(family="Helvetica", size=13),
            text_color=self.COLOR_MUTED,
        )
        subtitle_label.pack(pady=(0, 15))

        # ── status bar ──
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Your turn (X)",
            font=ctk.CTkFont(family="Helvetica", size=15, weight="bold"),
            text_color=self.COLOR_X,
        )
        self.status_label.pack(pady=(0, 12))

        # ── game board ──
        board_frame = ctk.CTkFrame(
            self.main_frame, fg_color=self.COLOR_GRID,
            corner_radius=12
        )
        board_frame.pack(padx=5, pady=5)

        self.buttons = []
        for i in range(9):
            row, col = divmod(i, 3)

            btn = ctk.CTkButton(
                board_frame,
                text="",
                width=110,
                height=110,
                font=ctk.CTkFont(family="Helvetica", size=40, weight="bold"),
                fg_color=self.COLOR_CARD,
                hover_color=self.COLOR_HOVER,
                text_color=self.COLOR_TEXT,
                corner_radius=8,
                command=lambda pos=i: self._human_move(pos),
            )
            btn.grid(row=row, column=col, padx=4, pady=4)
            self.buttons.append(btn)

        # ── scoreboard ──
        score_frame = ctk.CTkFrame(
            self.main_frame, fg_color=self.COLOR_BG
        )
        score_frame.pack(pady=(15, 5), fill="x")

        # wins
        win_frame = ctk.CTkFrame(score_frame, fg_color=self.COLOR_BG)
        win_frame.pack(side="left", expand=True)
        ctk.CTkLabel(
            win_frame, text="WINS",
            font=ctk.CTkFont(size=11), text_color=self.COLOR_MUTED
        ).pack()
        self.win_label = ctk.CTkLabel(
            win_frame, text="0",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=self.COLOR_WIN
        )
        self.win_label.pack()

        # draws
        draw_frame = ctk.CTkFrame(score_frame, fg_color=self.COLOR_BG)
        draw_frame.pack(side="left", expand=True)
        ctk.CTkLabel(
            draw_frame, text="DRAWS",
            font=ctk.CTkFont(size=11), text_color=self.COLOR_MUTED
        ).pack()
        self.draw_label = ctk.CTkLabel(
            draw_frame, text="0",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=self.COLOR_MUTED
        )
        self.draw_label.pack()

        # losses
        loss_frame = ctk.CTkFrame(score_frame, fg_color=self.COLOR_BG)
        loss_frame.pack(side="left", expand=True)
        ctk.CTkLabel(
            loss_frame, text="LOSSES",
            font=ctk.CTkFont(size=11), text_color=self.COLOR_MUTED
        ).pack()
        self.loss_label = ctk.CTkLabel(
            loss_frame, text="0",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=self.COLOR_X
        )
        self.loss_label.pack()

        # ── controls row 1: difficulty + new game ──
        controls_frame = ctk.CTkFrame(self.main_frame, fg_color=self.COLOR_BG)
        controls_frame.pack(pady=(12, 0), fill="x")

        # difficulty selector
        self.diff_menu = ctk.CTkSegmentedButton(
            controls_frame,
            values=["Easy", "Hard"],
            command=self._change_difficulty,
            font=ctk.CTkFont(size=12),
            selected_color=self.COLOR_O,
            selected_hover_color="#1a4a7a",
            unselected_color=self.COLOR_CARD,
            unselected_hover_color=self.COLOR_HOVER,
        )
        self.diff_menu.set("Hard")
        self.diff_menu.pack(side="left", padx=(0, 10))

        # new game button
        self.new_game_btn = ctk.CTkButton(
            controls_frame,
            text="New Game",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=self.COLOR_O,
            hover_color="#1a4a7a",
            width=120,
            height=32,
            corner_radius=8,
            command=self._new_game,
        )
        self.new_game_btn.pack(side="right")

        # ── controls row 2: learning toggle + info ──
        learn_frame = ctk.CTkFrame(self.main_frame, fg_color=self.COLOR_BG)
        learn_frame.pack(pady=(8, 0), fill="x")

        self.learn_switch = ctk.CTkSwitch(
            learn_frame,
            text="Live Learning",
            font=ctk.CTkFont(size=12),
            text_color=self.COLOR_LEARN,
            progress_color=self.COLOR_LEARN,
            button_color=self.COLOR_TEXT,
            button_hover_color="#ffffff",
            fg_color=self.COLOR_CARD,
            command=self._toggle_learning,
        )
        self.learn_switch.select()  # on by default
        self.learn_switch.pack(side="left")

        self.learn_info_label = ctk.CTkLabel(
            learn_frame,
            text=f"Q-table: {len(self.agent.q_table)} states",
            font=ctk.CTkFont(size=11),
            text_color=self.COLOR_MUTED,
        )
        self.learn_info_label.pack(side="right")

    def _new_game(self):
        """Reset the board and start a fresh game."""
        self.env.reset()
        self.game_active = True
        self.move_history = []

        # clear all buttons
        for btn in self.buttons:
            btn.configure(
                text="",
                fg_color=self.COLOR_CARD,
                text_color=self.COLOR_TEXT,
                state="normal",
            )

        self.status_label.configure(
            text="Your turn (X)", text_color=self.COLOR_X
        )

    def _human_move(self, position):
        """Handle a click on a board cell."""
        if not self.game_active:
            return

        # check if the cell is available
        if self.env.board[position] != 0:
            return

        # record the move BEFORE stepping (we need the pre-move state)
        state_before = self.env.get_state()

        self.move_history.append({
            "player": self.env.current_player,
            "state": state_before,
            "action": position,
        })

        # make the human's move
        self._place_mark(position, "X")
        state, reward, done, info = self.env.step(position)

        if done:
            self._handle_game_over(info)
            return

        # agent's turn — small delay so it feels natural
        self.game_active = False  # prevent clicks during agent's "thinking"
        self.status_label.configure(
            text="Agent thinking...", text_color=self.COLOR_O_TEXT
        )
        self.root.after(400, lambda: self._agent_move())

    def _agent_move(self):
        """Let the agent pick and play its move."""
        state = self.env.get_state()
        valid_actions = self.env.get_valid_actions()

        if self.difficulty == "easy":
            random_agent = RandomAgent()
            action = random_agent.choose_action(state, valid_actions)
        else:
            # use epsilon=0 for playing (no random exploration during gameplay)
            action = self.agent.choose_action(state, valid_actions, training=False)

        # record agent's move
        self.move_history.append({
            "player": self.env.current_player,
            "state": state,
            "action": action,
        })

        self._place_mark(action, "O")
        state, reward, done, info = self.env.step(action)

        if done:
            self._handle_game_over(info)
        else:
            self.game_active = True
            self.status_label.configure(
                text="Your turn (X)", text_color=self.COLOR_X
            )

    def _place_mark(self, position, symbol):
        """Update a button to show X or O."""
        if symbol == "X":
            color = self.COLOR_X
        else:
            color = self.COLOR_O_TEXT

        self.buttons[position].configure(
            text=symbol,
            text_color=color,
            state="disabled",
        )

    def _handle_game_over(self, info):
        """Display the result, update scores, and learn from the game."""
        self.game_active = False
        winner = info["winner"]

        # find and highlight the winning line if there is one
        winning_line = self._get_winning_line()

        if winner == 1:
            self.score["wins"] += 1
            self.status_label.configure(
                text="You won! 🎉", text_color=self.COLOR_WIN
            )
            if winning_line:
                self._highlight_line(winning_line, self.COLOR_WIN)

        elif winner == -1:
            self.score["losses"] += 1
            self.status_label.configure(
                text="Agent wins! 🤖", text_color=self.COLOR_X
            )
            if winning_line:
                self._highlight_line(winning_line, self.COLOR_X)

        else:
            self.score["draws"] += 1
            self.status_label.configure(
                text="It's a draw! 🤝", text_color=self.COLOR_MUTED
            )

        # update scoreboard
        self.win_label.configure(text=str(self.score["wins"]))
        self.draw_label.configure(text=str(self.score["draws"]))
        self.loss_label.configure(text=str(self.score["losses"]))

        # disable remaining buttons
        for btn in self.buttons:
            btn.configure(state="disabled")

        # ── LEARN FROM THIS GAME ──
        if self.learning_enabled and self.difficulty == "hard":
            self._learn_from_game(winner)

    def _learn_from_game(self, winner):
        """
        Update the agent's Q-table based on the game that just finished.

        We walk through the move history and assign rewards:
        - The agent's moves (player -1) get +1 for win, -1 for loss, +0.5 for draw
        - We update Q-values backwards so future info propagates correctly
        """
        if not self.move_history:
            return

        # only update from the agent's perspective (player -1, which is "O")
        agent_player = -1

        # figure out the agent's reward for this game
        if winner == agent_player:
            final_reward = 1.0   # agent won
        elif winner == 0:
            final_reward = 0.5   # draw
        else:
            final_reward = -1.0  # agent lost (human won)

        # collect agent's moves in order
        agent_moves = [m for m in self.move_history if m["player"] == agent_player]

        # update Q-values backwards (last move first)
        for i in reversed(range(len(agent_moves))):
            move = agent_moves[i]
            state = move["state"]
            action = move["action"]

            if i == len(agent_moves) - 1:
                # last move — use the final game reward
                self.agent.update(state, action, final_reward, state, [], done=True)
            else:
                # mid-game move — bootstrap from the next state
                next_move = agent_moves[i + 1]
                next_state = next_move["state"]
                next_valid = [j for j in range(9) if next_state[j] == 0]
                self.agent.update(state, action, 0.0, next_state, next_valid, done=False)

        # also learn from the human's moves (helps the agent understand both sides)
        human_player = 1
        human_moves = [m for m in self.move_history if m["player"] == human_player]

        if winner == human_player:
            human_final = 1.0
        elif winner == 0:
            human_final = 0.5
        else:
            human_final = -1.0

        for i in reversed(range(len(human_moves))):
            move = human_moves[i]
            state = move["state"]
            action = move["action"]

            if i == len(human_moves) - 1:
                self.agent.update(state, action, human_final, state, [], done=True)
            else:
                next_move = human_moves[i + 1]
                next_state = next_move["state"]
                next_valid = [j for j in range(9) if next_state[j] == 0]
                self.agent.update(state, action, 0.0, next_state, next_valid, done=False)

        # save updated Q-table
        self.agent.save(self.model_path)

        # update the Q-table size display
        self.learn_info_label.configure(
            text=f"Q-table: {len(self.agent.q_table)} states"
        )

    def _get_winning_line(self):
        """Return the indices of the winning line, or None."""
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6],
        ]
        for line in lines:
            vals = [self.env.board[i] for i in line]
            if vals[0] != 0 and vals[0] == vals[1] == vals[2]:
                return line
        return None

    def _highlight_line(self, line, color):
        """Highlight the winning cells."""
        for idx in line:
            self.buttons[idx].configure(fg_color=color, text_color="#ffffff")

    def _change_difficulty(self, value):
        """Switch between easy and hard mode."""
        self.difficulty = value.lower()
        self._new_game()

    def _toggle_learning(self):
        """Toggle live learning on/off."""
        self.learning_enabled = self.learn_switch.get() == 1

        if self.learning_enabled:
            self.learn_info_label.configure(
                text=f"Q-table: {len(self.agent.q_table)} states"
            )
        else:
            self.learn_info_label.configure(text="Learning paused")

    def run(self):
        """Start the application."""
        self.root.mainloop()


def main():
    app = TicTacToeGUI()
    app.run()


if __name__ == "__main__":
    main()
