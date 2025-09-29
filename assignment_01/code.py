import copy
import time
import numpy as np
from collections import deque, defaultdict

class Node:
    """
    Represents a node in the Tic-Tac-Toe game tree.
    """
    def __init__(self, name, state, player, is_win=False, is_draw=False):
        self.name = name
        self.state = state
        self.player = player
        self.is_win = is_win
        self.is_draw = is_draw


def is_win_position(grid, player):
    """
    Checks if the given player has a winning position on the board.
    """
    grid_size = len(grid)
    is_row_win = False
    is_col_win = False
    win_row_col = [player] * grid_size

    # Check for diagonal win
    is_diag_win = all(grid[i, i] == player for i in range(grid_size))
    is_other_diag_win = all(grid[i][grid_size - i - 1] == player for i in range(grid_size))
    # Check if a row or column win
    for i in range(grid_size):
        row = grid[i, :]
        col = grid[:, i]
        if all(row == win_row_col):
            is_row_win = True
        if all(col == win_row_col):
            is_col_win = True

    is_win = is_diag_win or is_other_diag_win or is_row_win or is_col_win
    return is_win


def is_draw(grid):
    """
    Checks if the game is a draw given the current board state.
    """
    is_win_x = is_win_position(grid, "x")
    is_win_o = is_win_position(grid, "o")
    is_draw = "-" not in grid

    return is_draw and not (is_win_x or is_win_o)


def generate_possible_combinations(grid, player):
    """
    Generates all possible next board states by placing the player's mark
    in every available position.
    """
    possible_states = []
    for idx, row in enumerate(grid):
        for jdx, col in enumerate(row):
            grid_copy = copy.deepcopy(grid)
            if col == "-":
                grid_copy[idx][jdx] = player
                possible_states.append(grid_copy)

    return possible_states


def build_tree(grid, player=None):
    """
    Builds the game tree starting from the given board state, alternating
    players for each level, until terminal states are reached.
    """
    counter = 0
    all_nodes = []
    tree = defaultdict(list)

    parent_node = Node(name=f"{counter}", state=grid, player=player)
    all_nodes.append(parent_node)
    counter += 1

    for node in all_nodes:
        if node.player is not None and is_win_position(node.state, node.player):
            node.is_win = True
            continue
        if node.player is not None and is_draw(node.state):
            node.is_draw = True
            continue

        if node.player == "x":
            cur_player = "o"
        else:
            cur_player = "x"
        possible_combs = generate_possible_combinations(node.state, player=cur_player)

        for possible_state in possible_combs:
            state_name = f"{counter}"
            child_node = Node(name=state_name, state=possible_state, player=f"{cur_player}")

            all_nodes.append(child_node)
            tree[f"{node.name}"].append(child_node.name)
            counter += 1

    name_to_node = {node.name: node for node in all_nodes}
    del all_nodes

    return tree, name_to_node, counter


def traverse_tree(root, name_to_node, algo="bfs", depth_limit=1):
    """
    Traverses the game tree using BFS, DFS, or Iterative Deepening DFS to find
    a winning path.
    """
    assert algo in ["bfs", "dfs", "idfs"], "algo parametre must be one of [bfs, dfs, idfs]"
    if algo == "idfs":
        assert depth_limit > 0, "depth limit must be greater than 0"
    assert root is not None
    visited_nodes = set()

    if algo == "bfs":
        queue = deque([("0", ["0"])])
        while queue:
            node_name, path = queue.popleft()
            visited_nodes.add(node_name)

            if name_to_node[node_name].is_win:
                return path, len(visited_nodes)

            for child_name in root[node_name]:
                if child_name not in visited_nodes:
                    queue.append((child_name, path + [child_name]))

    elif algo == "dfs":
        stack = [("0", ["0"])]
        while stack:
            node_name, path = stack.pop()
            visited_nodes.add(node_name)
            if name_to_node[node_name].is_win:
                return path, len(visited_nodes)

            for child_name in reversed(root[node_name]):
                if child_name not in visited_nodes:
                    stack.append((child_name, path + [child_name]))

    elif algo == "idfs":
        stack = [("0", ["0"])]
        while stack:
            node_name, path = stack.pop()
            visited_nodes.add(node_name)

            if name_to_node[node_name].is_win:
                return path, len(visited_nodes)

            if (len(path) - 1) == depth_limit:
                return None, len(visited_nodes)

            for child_name in reversed(root[node_name]):
                if child_name not in visited_nodes:
                    stack.append((child_name, path + [child_name]))

    return None


def main(grid, last_move_player):
    """
    Builds the game tree from the initial board state and performs BFS, DFS,
    and iterative deepening DFS traversals to find winning paths.
    """
    tree, name_to_node, total_states = build_tree(grid, player=last_move_player)

    print(f"Total states found: {total_states}")

    print("Traversing with bfs")

    start = time.time()
    path_to_win, num_nodes_visited = traverse_tree(tree, name_to_node, algo="bfs")
    end = time.time()

    print(f"bfs took: {end-start:.6f}s")
    print(f"bfs visited {num_nodes_visited} nodes")

    if path_to_win is None:
        print("No winning state found!")
    else:
        print(f"The win path: {path_to_win}")
        print("Optimal actions that lead to win: ")
        for n in path_to_win:
            print(name_to_node[n].state)
            print()
    print()

    print("Traversing with dfs")
    start = time.time()
    path_to_win, num_nodes_visited = traverse_tree(tree, name_to_node, algo="dfs")
    end = time.time()

    print(f"The win path: {path_to_win}")
    print(f"dfs took: {end-start:.6f}s")
    print(f"dfs visited {num_nodes_visited} nodes")

    if path_to_win is None:
        print("No winning state found within provided depth_limit")
    else:
        print(f"The win path: {path_to_win}")
        print("Optimal actions that lead to win: ")
        for n in path_to_win:
            print(name_to_node[n].state)
            print()
    print()

    print("Traversing with idfs")
    start = time.time()
    path_to_win, num_nodes_visited = traverse_tree(tree, name_to_node, algo="idfs", 
                                                   depth_limit=5)

    end = time.time()

    print(f"bfs took: {end-start:.6f}s")
    print(f"bfs visited {num_nodes_visited} nodes")

    if path_to_win is None:
        print("No winning state found within provided depth_limit")
    else:
        print(f"The win path: {path_to_win}")
        print("Optimal actions that lead to win: ")
        for n in path_to_win:
            print(name_to_node[n].state)
            print()

    return 0


if __name__ == "__main__":
    grid = np.array([
        ["-", "-", "-"],
        ["-", "-", "-"],
        ["-", "-", "-"]
    ])
    #print(is_win_position(grid, "x"))
    main(grid=grid, last_move_player="o")
