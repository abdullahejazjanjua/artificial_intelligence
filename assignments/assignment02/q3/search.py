"""Implementation of the A* algorithm.

This file contains a skeleton implementation of the A* algorithm. It is a single
method that accepts the root node and runs the A* algorithm
using that node's methods to generate children, evaluate heuristics, etc.
This way, plugging in root nodes of different types, we can run this A* to
solve different problems.

"""

def Astar(root):
    """Runs the A* algorithm given the root node. The class of the root node
    defines the problem that's being solved. The algorithm either returns the solution
    as a path from the start node to the goal node or returns None if there's no solution.

    Parameters
    ----------
    root: Node
        The start node of the problem to be solved.

    Returns
    -------
        path: list of Nodes or None
            The solution, a path from the initial node to the goal node.
            If there is no solution it should return None
    """

    frontier = []
    visited = set()
    frontier.append(root)

    while frontier:
        frontier.sort(key= lambda x: x.f)
        current_node = frontier.pop(0)
        if current_node.is_goal():
            return current_node.get_path()

        visited.add(current_node.state)
        for child_node in current_node.generate_children():
            if child_node.state not in visited:
                frontier.append(child_node)

    return []
