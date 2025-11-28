import time
from statistics import mean
from typing import Optional

import numpy as np

from node import Node

MAX_DEPTH: int = 5
PLAYER: int = 1 # AI
OPPONENT: int = 2 # Human

nodes_expanded: int = 0

def evaluation_function(node: Node):
    score = 0
    # Loop through all rows
    for i in range(node.board.shape[0]):
        # Loop through all columns
        for j in range(node.board.shape[1]):
            if node.board[i][j] == PLAYER:
                score += 1.0  # Simple heuristic: reward for owning a space
            elif node.board[i][j] == OPPONENT:
                score -= 10.0 # Simple heuristic: high penalty for opponent pieces
    return score


def minimax(node: Node, is_maximize: bool, depth: int, alpha: float = float('-inf'), beta: float = float('inf')) -> float:
    global nodes_expanded
    nodes_expanded += 1

    # Terminal check 1: AI wins, return high score
    if node.is_terminal(PLAYER):
        return 100000000.0
    # Terminal check 2: Opponent wins, return low score
    if node.is_terminal(OPPONENT):
        return -100000000.0
    # Terminal check 3: Draw, return neutral score
    if node.is_draw():
        return 0.0

    # Base case: Max depth reached, return heuristic score
    if depth == 0:
        return evaluation_function(node)

    # Maximizing Player's turn (AI)
    if is_maximize:
        max_eval = float('-inf')
        # Iterate over all available moves
        for (row_idx, col_idx) in node.available_moves():
            node.make_move(row_idx, col_idx, PLAYER) # Make move for the maximizing player
            eval = minimax(node, False, depth - 1, alpha, beta) # Recurse to minimizing layer
            node.undo_move(row_idx, col_idx) # Undo move
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    # Minimizing Player's turn (Human)
    else:
        min_eval = float('inf')
        # Iterate over all available moves
        for (row_idx, col_idx) in node.available_moves():
            node.make_move(row_idx, col_idx, OPPONENT) # make move
            eval = minimax(node, True, depth - 1, alpha, beta) # Recurse to maximizing layer
            node.undo_move(row_idx, col_idx) # Undo move (backtracking)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def best_move(root: Node) -> Optional[int]:
    best_score = float('-inf')
    move = None

    # Evaluate moves for the AI (PLAYER)
    for (row_idx, col_idx) in root.available_moves():
        root.make_move(row_idx, col_idx, PLAYER) # Move move
        score = minimax(root, False, MAX_DEPTH) # Initial call to Minimax
        root.undo_move(row_idx, col_idx) # Undo move

        # Track the move that yields the highest score
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

        # Find the correct row to place the piece
        while row_idx >= 0:
            if board[row_idx][col_idx] == 0:
                break
            row_idx -= 1

        if row_idx < 0:
            print("Column is full! Try again.")
            continue

        root.make_move(row_idx, col_idx, OPPONENT) # Human (Player 2) makes a move
        print("You made your move!")
        print_board(board)

        if root.is_terminal(OPPONENT):
            print("You win!")
            break
        if root.is_draw():
            print("It's a draw!")
            break

        print("AI is making its move!")
        start = time.time()
        ai_move = best_move(root) # AI determines optimal move
        end = time.time()
        time_taken.append((end-start))

        print(f"Nodes expanded: {nodes_expanded}")
        print(f"Time taken: {time_taken[-1]:.4f}s\n")

        # Find the correct row for the AI's chosen column
        row_idx = H - 1
        while row_idx >= 0:
            if board[row_idx][ai_move] == 0:
                break
            row_idx -= 1

        root.make_move(row_idx, ai_move, PLAYER) # AI (Player 1) makes a move
        print_board(board)

        if root.is_terminal(PLAYER):
            print("AI wins!")
            break
        if root.is_draw():
            print("It's a draw!")
            break


    print(f"Average Time taken per move: {mean(time_taken):.4f}s")

if __name__ == "__main__":
    grid = np.zeros((6, 7), dtype=int)
    play_game(grid)