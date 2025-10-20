def check_assignment_consistency(region, color):
    """
    Check if assignment violates the imposed constraint
    """
    for neighbour in regions[region]:
        if region in assignment and assignment[neighbour] == color:
            return False
    return True


def get_unassigned_region():
    """
    Finds and returns the first unassigned node
    """
    for region in regions:
        if region not in assignment:
            return region
    return None


def print_result(result):
    """
    Formats the assignment and prints it nicely
    """
    print("============")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("============\n")

def backtrack():
    """
    Implementation of Backtracking for assigning colors to regions with
    constraint color of node_1 != node_2
    """
    if len(assignment) == len(regions):
        return assignment

    unassigned_region = get_unassigned_region()
    for color in domain:
        if check_assignment_consistency(unassigned_region, color):
            assignment[unassigned_region] = color
            result = backtrack()
            if result:
                print_result(result)
            del assignment[unassigned_region]
    return None


if __name__ == "__main__":
    regions = {
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'Q', 'SA'],
        'Q': ['NT', 'NSW', 'SA'],
        'NSW': ['Q', 'SA', 'V'],
        'V': ['SA', 'NSW'],
        'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
        'T': ['SA']
    }

    domain = ['red', 'green', 'blue']
    assignment = {}

    backtrack()
