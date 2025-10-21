import copy

def check_constraint_consistency(grid, graph, cell, value):
    for (idx, jdx) in graph[cell]:
        if value == grid[idx][jdx]:
            return False
    return True

def get_unassigned_cell(grid, assignment, graph):
    for cell in graph:
        if cell not in assignment and grid[cell[0]][cell[1]] == 0:
            return cell
    return None

def create_grid(file_name: str) -> list[list[int]]:
    grid = []
    with open(file_name, "r") as f:
        data = f.read()

    for idx, line in enumerate(data.split("\n")):
        if idx == 0:
            continue
        else:
            row = []
            for row_element in line.split(" "):
                if row_element.isdigit():
                    row.append(int(row_element))
            if row:
                grid.append(row)

    return grid


def fill_grid(grid, assignment):
    grid_copy = copy.deepcopy(grid)
    for (idx, jdx), value in assignment.items():
        if grid[idx][jdx] == 0:
            grid_copy[idx][jdx] = value
    return grid_copy


def prefill_assignment(grid):
    grid_size = len(grid)
    assignment = {}
    for idx in range(grid_size):
        for jdx in range(grid_size):
            if grid[idx][jdx] != 0:
                assignment[(idx, jdx)] = grid[idx][jdx]
    return assignment


def create_constraint_graph(grid: list[list[int]], block_size):
    grid_size = len(grid)

    graph = {}
    for idx in range(grid_size):
        for jdx in range(grid_size):
            graph[(idx, jdx)] = []

    # Add row constraint
    for idx in range(grid_size):
        for jdx in range(grid_size):
            for k in range(grid_size):
                if k != jdx:
                    graph[(idx, jdx)].append((idx, k))


    # Add col constraint
    for idx in range(grid_size):
        for jdx in range(grid_size):
            for k in range(grid_size):
                if k != idx:
                    graph[(idx, jdx)].append((k, jdx))

    # Add block constraint
    for idx in range(grid_size):
        for jdx in range(grid_size):
            block_i = (idx // block_size) * block_size
            block_j = (jdx // block_size) * block_size
            for x in range(block_size):
                for y in range(block_size):
                    if (block_i + x, block_j + y) != (idx, jdx):
                       graph[(idx, jdx)].append((block_i + x, block_j + y))

    return graph


def backtrack(grid, assignment, graph, domain):
    if len(assignment) == len(graph):
        return assignment

    unassigned_cell = get_unassigned_cell(grid, assignment, graph)
    assert unassigned_cell, "There are no unassigned cells"

    for value in domain:
        if check_constraint_consistency(grid, graph, unassigned_cell, value):
                assignment[unassigned_cell] = value
                grid_copy = copy.deepcopy(grid)
                grid_copy[unassigned_cell[0]][unassigned_cell[1]] = value
                result = backtrack(grid_copy, assignment, graph, domain)
                if result:
                    return result
                del assignment[unassigned_cell]

    return None

def main(filename, block_size, max_value):
    grid = create_grid(filename)
    graph = create_constraint_graph(grid, block_size)
    assignment = prefill_assignment(grid)
    domain = [i for i in range(1, max_value + 1)]

    result = backtrack(grid, assignment, graph, domain)

    if result is None:
        print("No solution found!")
    solution = fill_grid(grid, assignment)
    print("The solution to given Sodoku is:")
    for row in solution:
        print(row)

    with open("solution.txt", "w") as f:
        for row in solution:
            for row_element in row:
                f.write(f"{row_element} ")
            f.write("\n")

    print("The solution is append to solution.txt")
if __name__ == "__main__":
    main(filename="problem.txt", block_size=3, max_value=9)
