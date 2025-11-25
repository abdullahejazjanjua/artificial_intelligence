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
                    # Check rows
                    if (jdx + 3 < W) and np.all([self.board[idx][jdx + k] == player for k in range(4)]):
                        return True
                    # Check columns
                    if idx + 3 < H and np.all([self.board[idx + k][jdx] == player for k in range(4)]):
                        return True
                    if idx + 3 < H and jdx + 3 < W  and np.all([self.board[idx + k][jdx + k] == player for k in range(4)]):
                        return True
                    if idx - 3 >= 0 and jdx + 3 < W and np.all([self.board[idx - k][jdx + k] == player for k in range(4)]):
                        return True
        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def make_move(self, idx:int, jdx:int, player:int):
        self.board[idx][jdx] = player

    def undo_move(self, idx:int, jdx:int):
            self.board[idx][jdx] = 0

    def available_moves(self) -> list[tuple[int, int]]:
        moves = []
        H, W = self.board.shape

        for col_idx in range(W):
            row_idx = H - 1
            while row_idx >= 0:
                if self.board[row_idx][col_idx] == 0:
                    moves.append((row_idx, col_idx))
                    break
                row_idx -= 1

        return moves
