import random
import math
import matplotlib.pyplot as plt

# -------------------------------
# Nairobi Area Coordinates
# -------------------------------
cities = {
    "CBD": (-1.286389, 36.817223),
    "Westlands": (-1.2649, 36.8110),
    "Karen": (-1.3361, 36.7200),
    "Ngong": (-1.3530, 36.6570),
    "Kasarani": (-1.2195, 36.8960),
    "Lavington": (-1.2805, 36.7802),
    "Gigiri": (-1.2290, 36.8170),
    "South B": (-1.3100, 36.8500)
}

# -------------------------------
# Distance Formula (Euclidean)
# -------------------------------
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def total_distance(tour):
    dist = 0
    for i in range(len(tour)):
        dist += distance(cities[tour[i]], cities[tour[(i + 1) % len(tour)]])
    return dist

# -------------------------------
# Create a Neighbor Solution
# -------------------------------
def neighbor(tour):
    new_tour = tour.copy()
    i, j = random.sample(range(len(tour)), 2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

# -------------------------------
# Simulated Annealing Algorithm
# -------------------------------
def simulated_annealing(tour, initial_temp, cooling_rate, max_iter):
    current_tour = tour
    current_dist = total_distance(current_tour)
    best_tour = current_tour
    best_dist = current_dist
    T = initial_temp

    for i in range(max_iter):
        new_tour = neighbor(current_tour)
        new_dist = total_distance(new_tour)
        delta = new_dist - current_dist

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_tour = new_tour
            current_dist = new_dist

        if current_dist < best_dist:
            best_tour = current_tour
            best_dist = current_dist

        T *= cooling_rate

    return best_tour, best_dist

# -------------------------------
# Run the Algorithm
# -------------------------------
initial_tour = list(cities.keys())
random.shuffle(initial_tour)

best_tour, best_dist = simulated_annealing(
    initial_tour,
    initial_temp=10000,
    cooling_rate=0.999,
    max_iter=50000
)

print("Best Tour:", best_tour)
print("Best Distance Estimate:", best_dist)

# -------------------------------
# Visualization
# -------------------------------
def plot_tour(tour):
    x = [cities[city][1] for city in tour] + [cities[tour[0]][1]]
    y = [cities[city][0] for city in tour] + [cities[tour[0]][0]]

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, 'o-', markersize=10)
    for city in tour:
        plt.text(cities[city][1] + 0.002, cities[city][0] + 0.002, city)

    plt.title("TSP Optimized Route Across Nairobi using Simulated Annealing")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)

    # Save instead of show
    plt.savefig("nairobi_tsp_route.png", dpi=300, bbox_inches='tight')
    print("Map saved as nairobi_tsp_route.png")

