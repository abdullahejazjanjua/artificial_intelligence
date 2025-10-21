import queue
import math


def gbfs(start_node, goal_node):
    """
    Traverses graph using greedy best first search
    """
    pqueue = queue.PriorityQueue()
    visited = set()

    pqueue.put([heuristics[start_node], start_node, [start_node], 0])
    while not pqueue.empty():
        _, current_node, path, path_cost = pqueue.get()
        if current_node == goal_node:
            return path, path_cost

        visited.add(current_node)
        for neighbour, edge_cost in graph[current_node]:
            if neighbour not in visited:
                pqueue.put([heuristics[neighbour], neighbour, path + [neighbour], (path_cost + int(edge_cost))])
    return None, None


def a_star(start_node, goal_node):
    """
    Traverse graph using A star search
    """
    pqueue = queue.PriorityQueue()
    visited = set()
    cheapest_node_cost = {}

    pqueue.put([(heuristics[start_node] + 0), start_node, [start_node], 0])
    cheapest_node_cost[start_node] = 0
    while not pqueue.empty():
        _, current_node, path, path_cost = pqueue.get()
        if current_node == goal_node:
            return path, path_cost

        visited.add(current_node)
        for neighbour, edge_cost in graph[current_node]:
            current_node_cost = path_cost + int(edge_cost)
            if neighbour not in visited or current_node_cost < cheapest_node_cost[neighbour]:
                cheapest_node_cost[neighbour] = current_node_cost
                pqueue.put([(heuristics[neighbour] + current_node_cost), neighbour, path + [neighbour], current_node_cost])
    return None, None



def cal_heuristics(coordinates, goal_node):
    """
    Calculate heuristics for a goal_node
    """
    h_n = {}
    goal_x, goal_y = coordinates[goal_node]
    for node, (n_x, n_y) in coordinates.items():
        h_n[node] = math.sqrt(((n_x - goal_x)**2) + ((n_y - goal_y)**2))
    return h_n


if __name__ == "__main__":
    graph = {
        'A': [('B', 8), ('D', 3), ('F', 6)],
        'B': [('A', 8), ('C', 3), ('D', 2)],
        'C': [('B', 3), ('E', 5)],
        'D': [('A', 3), ('B', 2), ('C', 1), ('E', 8), ('G', 7)],
        'E': [('C', 5), ('D', 8), ('I', 5), ('J', 3)],
        'F': [('A', 6), ('G', 1), ('H', 7)],
        'G': [('D', 7), ('F', 1), ('I', 1)],
        'H': [('F', 7), ('I', 2)],
        'I': [('E', 5), ('G', 1), ('H', 2), ('J', 3)],
        'J': [('E', 3), ('I', 3)]
    }
    coordinates = {
        'A': (0, 0), 'B': (2, 1), 'C': (4, 1), 'D': (1, -2), 'E': (6, 0),
        'F': (-1, -3), 'G': (2, -4), 'H': (0, -6), 'I': (4, -5), 'J': (7, -3)
    }
    g_node = "I"
    s_node = "A"

    heuristics = cal_heuristics(coordinates, g_node)
    a_star_p, a_star_pcost = a_star(s_node, g_node)
    gbfs_p, gbfs_pcost = gbfs(s_node, g_node)
    print("Heuristics calculated:")
    print("node = heuristic")
    for n, h_n in heuristics.items():
        print(f"{n} = {h_n}")
    print("\nA star search")
    print(f"path: {a_star_p}")
    print(f"cost: {a_star_pcost}")
    print("\nGreedy best first search")
    print(f"path: {gbfs_p}")
    print(f"cost: {gbfs_pcost}")
