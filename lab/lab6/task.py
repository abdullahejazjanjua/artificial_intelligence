def check_assignment_consistency(node, assigned_value):
    """
    Checks if assignment doesn't violate the imposed constraint
    X1 != X2
    """
    for neighbour in graph[node]:
        if neighbour in assignment and assignment[neighbour] == assigned_value:
            return False
    return True


def get_unassigned_node():
    """
    Finds and returns the first unassigned node
    """
    for node in graph:
        if node not in assignment:
            return node
    return None


def backtrack():
    """
    Implements backtrack algorithm for assigning within constraint
    """
    if len(assignment) == len(graph):
        return assignment

    unassigned_node = get_unassigned_node()
    if unassigned_node is None:
        return

    for domain_i in domains[unassigned_node]:
        if check_assignment_consistency(unassigned_node, domain_i):
            assignment[unassigned_node] = domain_i
            result = backtrack()
            if result:
                print_result(result)
            del assignment[unassigned_node]
    return None


def print_result(result):
    """
    Formats and prints the result nicely
    """
    print("=============")
    for key, value in result.items():
        print(f"{key} : {value}")
    print("=============\n")


if __name__ == "__main__":
    graph = {
                "X1": ["X2"],
                "X2": ["X1"]
            }
    domains = {
                "X1": [1, 2],
                "X2": [2, 3]
              }
    assignment = {}
    backtrack()
