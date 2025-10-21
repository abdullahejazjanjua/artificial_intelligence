import copy

def fill_grid(grid, assignment):
    grid_copy = copy.deepcopy(grid)
    for (idx, jdx), value in assignment.items():
        grid_copy[idx][jdx] = value
    return grid_copy

def get_unassigned_row(grid, assignment):
    for row_idx in range(len(grid)):
        no_queen = True
        for col_idx in range(len(grid)):
            if grid[row_idx][col_idx] == 1:
                no_queen = False
        if no_queen:
            return row_idx
    return None

def check_constraint_consistency(graph, grid, cell, value):
    for neighbour in graph[cell]:
        if grid[neighbour[0]][neighbour[1]] == value:
            return False
    return True

def create_constraint_graph(grid):
    graph = {}
    grid_size = len(grid)

    for idx in range(grid_size):
        for jdx in range(grid_size):
            graph[(idx, jdx)] = []

    # Add row constraint
    for idx in range(grid_size):
        for jdx in range(grid_size):
            for k in range(grid_size):
                if k != jdx:
                   graph[(idx, jdx)].append((idx, k))

    # Add column constraint
    for idx in range(grid_size):
        for jdx in range(grid_size):
            for k in range(grid_size):
                if k != idx:
                   graph[(idx, jdx)].append((k, jdx))

    # Add main diagonal constraint
    for idx in range(grid_size):
        for jdx in range(grid_size):
            difference = idx - jdx
            for x in range(grid_size):
                for y in range(grid_size):
                    if ((x, y) != (idx, jdx)) and (x - y == difference):
                        graph[(idx, jdx)].append((x, y))

    # Add secondary diagonal constraint
    for idx in range(grid_size):
        for jdx in range(grid_size):
            sum = idx + jdx
            for x in range(grid_size):
                for y in range(grid_size):
                    if ((x, y) != (idx, jdx)) and (x + y == sum):
                        graph[(idx, jdx)].append((x, y))

    return graph

def backtrack(graph, grid, assignment):

    unassigned_row = get_unassigned_row(grid=grid, assignment=assignment)
    if unassigned_row is None:
        return assignment

    for col_idx in range(len(grid)):
        if check_constraint_consistency(graph=graph,grid=grid, cell=(unassigned_row, col_idx), value=1):
            assignment[(unassigned_row, col_idx)] = 1
            grid_copy = copy.deepcopy(grid)
            grid_copy[unassigned_row][col_idx] = 1

            result = backtrack(graph, grid_copy, assignment)
            if result:
                return result

            del assignment[(unassigned_row, col_idx)]


    return None


def main():
    grid = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]
    ]
    graph = create_constraint_graph(grid)
    assignment = {}
    result = backtrack(graph=graph, grid=grid, assignment=assignment)
    if not result:
        print("No solution found!")
        exit()

    solution = fill_grid(grid=grid, assignment=assignment)
    for row in solution:
        print(row)

if __name__ == "__main__":
    main()
