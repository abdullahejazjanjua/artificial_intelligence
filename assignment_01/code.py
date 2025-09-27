# for a given board state,
# Make a new board for all possible locations x could be placed.
# for each of these new boards,
# Make a new board o could be placed in.
# Repeat till you reach a terminal state
# Build a tree which I can search for optimal next placement for x.


import copy
import numpy as np


def is_win_position(grid, player):
    grid_size = len(grid)
    is_row_win = False
    is_col_win = False
    win_row_col = [player] * grid_size

    # Check for diagonal win
    is_diag_win = all(grid[i, i] == player for i in range(grid_size))
    is_other_diag_win = all(grid[i][grid_size - i - 1] ==
                            player for i in range(grid_size))
    # Check if a row or column win
    for i in range(grid_size):
        row = grid[i, :]
        col = grid[:, i]
        if all(row == win_row_col):
            is_row_win = True
        if all(col == win_row_col):
            is_col_win = True

    return is_diag_win | is_other_diag_win | is_row_win | is_col_win


def generate_possible_combinations(grid, player):
    for idx, row in enumerate(grid):
        for jdx, col in enumerate(row):
            grid_copy = copy.deepcopy(grid)
            if col == "-":
                grid_copy[idx][jdx] = player
                yield grid_copy


def define_tree():
    pass


def main(grid):
    result = generate_possible_combinations(grid, "x")
    for r in result:
        print(r)
        print()


if __name__ == "__main__":
    grid = np.array([
            ["x", "x", "x"],
            ["-", "-", "-"],
            ["-", "-", "-"]
        ])
    print(is_win_position(grid, "x"))
    # main(grid=grid)
