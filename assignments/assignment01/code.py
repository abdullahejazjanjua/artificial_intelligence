import copy
import time
from collections import deque, defaultdict

import numpy as np

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
    is_diag_win = np.all([grid[i, i] == player for i in range(grid_size)])
    is_other_diag_win = np.all([grid[i][grid_size - i - 1] == player for i in range(grid_size)])
    
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
    draw = not np.any(grid == "-")

    return draw and not (is_win_x or is_win_o)


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
                grid_copy[idx][jdx] = player # avoid modifying the original grid
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
    counter += 1 # Serves as a unique id for each node

    for node in all_nodes:
        
        if node.player is not None and is_win_position(node.state, node.player):
            node.is_win = True
            continue

        if node.player is not None and is_draw(node.state):
            node.is_draw = True
            continue

       # Alternate between players. If no last player, then set x as the player 
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


def bfs(root, name_to_node):
    """
    Performs bfs on the on the game tree returns the path to win state.
    """
    queue = deque([("0", ["0"])])
    visited_nodes = set()
    while queue:
        node_name, path = queue.popleft()
        visited_nodes.add(node_name)

        if name_to_node[node_name].is_win:
            return path, len(visited_nodes)

        for child_name in root[node_name]:
            if child_name not in visited_nodes:
                queue.append((child_name, path + [child_name]))
    return None, None


def dfs(root, name_to_node):
    """
    Performs DFS on the game tree and returns the path to win state.
    """
    stack = [("0", ["0"])]
    visited_nodes = set()
    while stack:
        node_name, path = stack.pop()
        visited_nodes.add(node_name)
        if name_to_node[node_name].is_win:
            return path, len(visited_nodes)

        for child_name in reversed(root[node_name]):
            if child_name not in visited_nodes:
                stack.append((child_name, path + [child_name]))

    return None, None


def iterative_deepening(root, name_to_node, depth_limit=1):
    """
    Performs Iterative Depening on the game tree and returns a path
    to win state
    """
    stack = [("0", ["0"])]
    visited_nodes = set()
    while stack:
        node_name, path = stack.pop()
        visited_nodes.add(node_name)

        if name_to_node[node_name].is_win:
            return path, len(visited_nodes)

        if (len(path) - 1) == depth_limit: # O-indexed depth
            return None, len(visited_nodes)

        for child_name in reversed(root[node_name]):
            if child_name not in visited_nodes:
                stack.append((child_name, path + [child_name]))
    return None, None


def print_stats(method_name, name_to_node, num_nodes_visited,
                path_to_win, time_taken):
    """
    Prints the stats of a provided method
    """
    print(f"{method_name} took: {time_taken:.6f}s")
    print(f"{method_name} visited {num_nodes_visited} nodes")

    if path_to_win is None:
        print("No winning state found!")
    else:
        print(f"The win path: {path_to_win}")
        print("Optimal actions that lead to win: ")
        for node_name in path_to_win:
            print(name_to_node[node_name].state)
            print()
    print()


def main(grid, last_move_player=None, num_max_depth=8):
    """
    Builds the game tree from the initial board state and performs BFS, DFS,
    and iterative deepening DFS traversals to find winning paths.
    """
    tree, name_to_node, total_states = build_tree(grid, last_move_player)

    print(f"Total states found: {total_states}")
    
    print("Traversing with BFS")
    start = time.time()
    path_to_win, num_nodes_visited = bfs(tree, name_to_node)
    end = time.time()
    print_stats("bfs", name_to_node, num_nodes_visited,
                path_to_win, (end-start))

    print("Traversing with DFS")
    start = time.time()
    path_to_win, num_nodes_visited = dfs(tree, name_to_node)
    end = time.time()
    print_stats("dfs", name_to_node, num_nodes_visited,
                path_to_win, (end-start))
    
    print("Traversing with Iterative Deepening")
    for depth in range(num_max_depth):
        print(f"Trying depth: {depth}")
        start = time.time()
        path_to_win, num_nodes_visited = \
            iterative_deepening(tree, name_to_node, depth_limit=depth)
        end = time.time()
        if path_to_win is not None:
            print_stats("iterative Deepening", name_to_node, num_nodes_visited,
                        path_to_win, (end-start))
            break

    return 0


if __name__ == "__main__":
    print("===================================")
    print("Using empty tic-tac-toe Grid")
    print("===================================")
    grid = np.array([
        ["-", "-", "-"],
        ["-", "-", "-"],
        ["-", "-", "-"]
    ])
    main(grid=grid)
    print()


    print("========================================")
    print("Using partially filled tic-tac-toe Grid")
    print("========================================")
    grid = np.array([
        ["x", "-", "x"],
        ["-", "o", "-"],
        ["-", "-", "-"]
    ])
    main(grid=grid, last_move_player="x")
    print()
