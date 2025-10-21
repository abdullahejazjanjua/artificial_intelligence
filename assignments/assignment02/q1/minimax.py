import time
from statistics import mean
from typing import Optional

import numpy as np

from node import Node

nodes_expanded: int = 0
def minimax(node: Node, is_maximize: bool, alpha: float = float('-inf'), beta: float = float('inf')) -> float:
    global nodes_expanded
    nodes_expanded += 1
    player: int = 1
    opponent: int = 2

    if node.is_terminal(player):
        return 1.0
    if node.is_terminal(opponent):
        return -1.0
    if node.is_draw():
        return 0.0

    if is_maximize:
        max_eval = float('-inf')
        for (row_idx, col_idx) in node.available_moves():
            node.make_move(row_idx, col_idx, opponent)
            eval = minimax(node, False, alpha, beta)
            node.undo_move(row_idx, col_idx)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for (row_idx, col_idx) in node.available_moves():
            node.make_move(row_idx, col_idx, opponent)
            eval = minimax(node, True, alpha, beta)
            node.undo_move(row_idx, col_idx)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def best_move(root: Node) -> Optional[int]:
    best_score = float('-inf')
    move = None

    for (row_idx, col_idx) in root.available_moves():
        root.make_move(row_idx, col_idx, 2)
        score = minimax(root, False)
        root.undo_move(row_idx, col_idx)
        if score > best_score:
            best_score = score
            move = col_idx
    return move


def print_board(board: np.ndarray) -> None:
    """Helper function to print the current board."""
    for row in board:
        print(' '.join(str(cell) for cell in row))
    print("\n")

def play_game(board: np.ndarray) -> None:
    H, W = board.shape
    root = Node(parent=None, board=board)

    print("Welcome to Connect 4!")
    print_board(board)
    time_taken = []
    while True:
        col_idx: int = int(input(f"Enter your move (column number (0 - {W - 1})): "))
        row_idx = H - 1

        while row_idx >= 0:
            if board[row_idx][col_idx] == 0:
                break
            row_idx -= 1

        if row_idx < 0:
            print("Column is full! Try again.")
            continue

        root.make_move(row_idx, col_idx, 2)
        print("You made your move!")
        print_board(board)

        if root.is_terminal(2):
            print("You win!")
            break
        if root.is_draw():
            print("It's a draw!")
            break

        print("AI is making its move!")
        start = time.time()
        ai_move = best_move(root)
        end = time.time()
        time_taken.append((end-start))
        print(f"Nodes expanded: {nodes_expanded}")
        print(f"Time taken: {time_taken[-1]:.4f}s\n")

        row_idx = H - 1
        while row_idx >= 0:
            if board[row_idx][ai_move] == 0:
                break
            row_idx -= 1

        root.make_move(row_idx, ai_move, 1)
        print_board(board)

        if root.is_terminal(1):
            print("AI wins!")
            break
        if root.is_draw():
            print("It's a draw!")
            break


    print(f"Average Time taken per move: {mean(time_taken):.4f}s")

if __name__ == "__main__":
    grid = np.zeros((6, 7), dtype=int)

    play_game(grid)
