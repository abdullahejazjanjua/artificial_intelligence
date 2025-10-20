import queue


def gbfs(start_node, goal_node):
    """
    Traverse a graph using greedy best first search
    """
    pqueue = queue.PriorityQueue()
    visited = set()

    # h(n), n, path, path_cost
    pqueue.put([hueristics[start_node], start_node, [start_node], 0])

    while not pqueue.empty():
        _, current_node, path, path_cost = pqueue.get()

        if current_node == goal_node:
            return path, path_cost

        visited.add(current_node)

        for neighbour, edge_cost in graph[current_node]:
            if neighbour not in visited:
                pqueue.put([hueristics[neighbour], neighbour,
                            path + [neighbour], (path_cost + int(edge_cost))])
    return None, None


if __name__ == "__main__":
    hueristics = {
        'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Dobreta': 242,
        'Eforie': 161, 'Fagaras': 178, 'Giurgiu': 77, 'Hirsova': 151,
        'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234,
        'Oradea': 380, 'Pitesti': 98, 'Rimnicu_Vilcea': 193, 'Sibiu': 253,
        'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
    }

    graph = {
        'Arad': [['Sibiu', 140], ['Timisoara', 118], ['Zerind', 75]],
        'Sibiu': [['Arad', 140], ['Fagaras', 99], ['Oradea', 151], ['Rimnicu_Vilcea', 80]],
        'Timisoara': [['Arad', 118], ['Lugoj', 111]],
        'Zerind': [['Arad', 75], ['Oradea', 71]],
        'Bucharest': [['Fagaras', 211], ['Giurgiu', 90], ['Pitesti', 101], ['Urziceni', 85]],
        'Fagaras': [['Bucharest', 211], ['Sibiu', 99]],
        'Giurgiu': [['Bucharest', 90]],
        'Pitesti': [['Bucharest', 101], ['Craiova', 138], ['Rimnicu_Vilcea', 97]],
        'Urziceni': [['Bucharest', 85], ['Hirsova', 98], ['Vaslui', 142]],
        'Craiova': [['Dobreta', 120], ['Pitesti', 138], ['Rimnicu_Vilcea', 146]],
        'Dobreta': [['Craiova', 120], ['Mehadia', 75]],
        'Mehadia': [['Dobreta', 75], ['Lugoj', 70]],
        'Lugoj': [['Mehadia', 70], ['Timisoara', 111]],
        'Oradea': [['Sibiu', 151], ['Zerind', 71]],
        'Rimnicu_Vilcea': [['Craiova', 146], ['Pitesti', 97], ['Sibiu', 80]],
        'Hirsova': [['Eforie', 86], ['Urziceni', 98]],
        'Eforie': [['Hirsova', 86]],
        'Iasi': [['Neamt', 87], ['Vaslui', 92]],
        'Neamt': [['Iasi', 87]],
        'Vaslui': [['Iasi', 92], ['Urziceni', 142]]
    }

    p, pcost = gbfs("Arad", "Bucharest")
    print(f"Path: {p}")
    print(f"cost: {pcost}")
