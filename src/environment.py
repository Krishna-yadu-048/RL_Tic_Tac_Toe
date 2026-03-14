import numpy as np


class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        self.done = False
        self.winner = None

    def reset(self):
        """Reset the board and return the initial state."""
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        """Return current board state as a tuple (hashable for Q-table)."""
        return tuple(self.board)

    def get_valid_actions(self):
        """Return list of empty positions."""
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        """
        Place current player's mark at the given position.

        Returns:
            next_state: board state after the move
            reward: reward for the current player
            done: whether the game is over
            info: dict with extra info (winner, etc.)
        """
        if self.done:
            raise ValueError("Game is already over. Call reset().")

        if self.board[action] != 0:
            raise ValueError(f"Position {action} is already taken.")

        # place the mark
        self.board[action] = self.current_player

        # check if this move won the game
        if self._check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif len(self.get_valid_actions()) == 0:
            # board is full, it's a draw
            self.done = True
            self.winner = 0
            reward = 0.5
        else:
            reward = 0.0

        info = {"winner": self.winner}
        next_state = self.get_state()

        # switch players
        self.current_player *= -1

        return next_state, reward, self.done, info

    def _check_win(self, player):
        """Check if the given player has three in a row."""
        b = self.board
        # all possible winning lines
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
            [0, 4, 8], [2, 4, 6]               # diagonals
        ]
        for line in lines:
            if all(b[i] == player for i in line):
                return True
        return False

    def render(self):
        """Print the board in a readable format."""
        symbols = {0: ".", 1: "X", -1: "O"}
        for row in range(3):
            cells = []
            for col in range(3):
                cells.append(symbols[self.board[row * 3 + col]])
            print(" | ".join(cells))
            if row < 2:
                print("---------")
        print()
