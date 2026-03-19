# Romanian Map represented as an adjacency list
romania_map = {
    'Arad': ['Zerind', 'Sibiu', 'Timisoara'],
    'Zerind': ['Arad', 'Oradea'],
    'Oradea': ['Zerind', 'Sibiu'],
    'Sibiu': ['Arad', 'Oradea', 'Fagaras', 'Rimnicu Vilcea'],
    'Timisoara': ['Arad', 'Lugoj'],
    'Lugoj': ['Timisoara', 'Mehadia'],
    'Mehadia': ['Lugoj', 'Drobeta'],
    'Drobeta': ['Mehadia', 'Craiova'],
    'Craiova': ['Drobeta', 'Rimnicu Vilcea', 'Pitesti'],
    'Rimnicu Vilcea': ['Sibiu', 'Craiova', 'Pitesti'],
    'Fagaras': ['Sibiu', 'Bucharest'],
    'Pitesti': ['Rimnicu Vilcea', 'Craiova', 'Bucharest'],
    'Bucharest': ['Fagaras', 'Pitesti', 'Giurgiu', 'Urziceni'],
    'Giurgiu': ['Bucharest'],
    'Urziceni': ['Bucharest', 'Hirsova', 'Vaslui'],
    'Hirsova': ['Urziceni', 'Eforie'],
    'Eforie': ['Hirsova'],
    'Vaslui': ['Urziceni', 'Iasi'],
    'Iasi': ['Vaslui', 'Neamt'],
    'Neamt': ['Iasi']
}

# Depth-First Search (Stack-Based)
def dfs_iterative(graph, start, goal):
    # Stack holds tuples: (current_city, path_so_far)
    stack = [(start, [start])]

    while stack:
        (city, path) = stack.pop()

        # Goal check
        if city == goal:
            return path

        # Push neighbors to stack
        for neighbor in graph[city]:
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))

    return None  # No path found

# Main program
if __name__ == "__main__":
    start_city = "Arad"
    goal_city = "Bucharest"

    print(f"Searching for a path from {start_city} to {goal_city}...\n")

    result = dfs_iterative(romania_map, start_city, goal_city)

    if result:
        print("Path found:")
        print(" → ".join(result))
    else:
        print("No path found.")
