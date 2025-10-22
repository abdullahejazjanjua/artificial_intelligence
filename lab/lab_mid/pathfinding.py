import queue
import copy


def calculate_hueristic(grid: list[list[str]], goal_position: tuple[int, int]):
    H, W = len(grid), len(grid[0])
    for idx in range(H):
        for jdx in range(W):
            if grid[idx][jdx] == 'S':
                return (abs(goal_position[0] - idx) + abs(goal_position[1] - jdx))
    return -1


def is_goal(grid: list[list[str]], goal_position: tuple[int, int]):
    return grid[goal_position[0]][goal_position[1]] == 'S'

def generate_children(grid: list[list[str]]):
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    H, W = len(grid), len(grid[0])
    children = []
    for idx in range(H):
        for jdx in range(W):
            if grid[idx][jdx] == 'S':
                for (row_inc, col_inc) in moves:
                    row_idx = idx + row_inc
                    col_idx = jdx + col_inc
                    if (0 <= row_idx < H) and (0 <= col_idx < W):
                        if grid[row_idx][col_idx] == '.':
                            updated_grid = copy.deepcopy(grid)
                            updated_grid[idx][jdx] = '.'
                            updated_grid[row_idx][col_idx] = 'S'
                            children.append(updated_grid)
    return children

def to_tuple(grid: list[list[str]]):
    return tuple(tuple(row) for row in grid)

def A_star(grid: list[list[str]], goal_position: tuple[int, int]):
    pqueue = queue.PriorityQueue()
    visited = set()
    cheapest_node_so_far = {}
    pqueue.put(
        [(calculate_hueristic(grid, goal_position) + 0), grid, [grid], 0]
    )
    cheapest_node_so_far[to_tuple(grid)] = 0

    while not pqueue.empty():
        _, current_grid, path, path_cost = pqueue.get()
        if is_goal(current_grid, goal_position):
            return path, path_cost

        visited.add(to_tuple(current_grid))
        for child_grid in generate_children(current_grid):
            current_node_cost = path_cost + 1
            if to_tuple(child_grid) not in visited or current_node_cost < cheapest_node_so_far[to_tuple(grid)]:
                pqueue.put(
                    [(calculate_hueristic(child_grid, goal_position) + current_node_cost), child_grid, path + [child_grid], current_node_cost]
                )

    return None, None

if __name__ == "__main__":
    grid = [
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '#', '#', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '#', '.', '.', '.', '.'],
        ['.', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
        ['.', 'S', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '#', '#', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
    ]

    path, path_cost = A_star(grid, goal_position=(9, 6))
    if path is None:
        print("solution not found!")
        exit()

    for g in path:
        for row in g:
            print(row)
        print()
    print(f"Total path cost: {path_cost}")
