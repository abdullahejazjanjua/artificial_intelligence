from typing import Optional
import numpy as np

class Node:
    def __init__(self, parent: Optional['Node'], board: np.ndarray):
        self.parent = parent
        self.board = board

    def is_terminal(self, player):
        H, W = self.board.shape

        for idx in range(H):
            for jdx in range(W):
                if self.board[idx][jdx] == player:
                    # Check horizontal (rows) for 4 connected pieces
                    if (jdx + 3 < W) and np.all([self.board[idx][jdx + k] == player for k in range(4)]):
                        return True
                    # Check vertical (columns) for 4 connected pieces
                    if idx + 3 < H and np.all([self.board[idx + k][jdx] == player for k in range(4)]):
                        return True
                    # Check positive diagonal
                    if idx + 3 < H and jdx + 3 < W  and np.all([self.board[idx + k][jdx + k] == player for k in range(4)]):
                        return True
                    # Check negative diagonal
                    if idx - 3 >= 0 and jdx + 3 < W and np.all([self.board[idx - k][jdx + k] == player for k in range(4)]):
                        return True
        return False

    def is_draw(self):
        # Return true if no empty zeros remain on the board
        return np.all(self.board != 0)

    def make_move(self, idx:int, jdx:int, player:int):
        self.board[idx][jdx] = player

    def undo_move(self, idx:int, jdx:int):
        # Reset cell to 0 to backtrack state
        self.board[idx][jdx] = 0

    def available_moves(self) -> list[tuple[int, int]]:
        moves = []
        H, W = self.board.shape

        for col_idx in range(W):
            # Start search from bottom row (H-1) due to gravity mechanics
            row_idx = H - 1
            while row_idx >= 0:
                if self.board[row_idx][col_idx] == 0:
                    moves.append((row_idx, col_idx))
                    break
                row_idx -= 1

        return moves