import random

# Define the problem instance (i.e. the distances between cities)
import random
import math

# Define the parameters of the problem
num_ants = 10
alpha = 1.0
beta = 2.0
evaporation = 0.5
q0 = 0.9
num_iterations = 100



def generate_cities(num_cities=5, pheromone_initial=1, max_distance=100):
    cities = []
    for i in range(num_cities):
        x = random.uniform(0, max_distance)
        y = random.uniform(0, max_distance)
        cities.append((x, y))
    #print(cities)
     
    pheromone = [[pheromone_initial for j in range(num_cities)] for i in range(num_cities)]

    distances = []
    for i in range(num_cities):
        row = []
        for j in range(num_cities):
            if i == j:
                row.append(0)
            else:
                xi, yi = cities[i]
                xj, yj = cities[j]
                distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                row.append(distance)
        distances.append(row)
        
    return distances, pheromone, cities



generate_cities()

# # Define the ant class
# class Ant:
#     def __init__(self, start_city):
#         self.v_cities = [start_city]
#         self.total_distance = 0.0
    
#     def select_next_city(self):
#         probs = []
#         c_city = self.v_cities[-1] #select current city by selecting the last city of visited
#         total_pheromone = 0.0
#         for city in range(len(distances)):
#             if city not in self.v_cities:
#                 pheromone_level = pheromone[c_city][city]
#                 distance = distances[c_city][city]
#                 probability = (pheromone_level ** alpha) * ((1.0 / distance) ** beta)
#                 probs.append((city, probability))
#                 total_pheromone += probability
#                 #(T^alpha*(1/d)^beta)/(Sum(T^alpha*(1/d)^beta))

#         if random.random() < q0: # (exploitation)
#             probs.sort(key=lambda x: x[1], reverse=True)
#             next_city = probs[0][0]
#         else: # (exploration)
#             probs = [(c, p / total_pheromone) for c, p in probs]
#             next_city = random.choices([c for c, p in probs], weights=[p for c, p in probs])[0]
        
#         self.v_cities.append(next_city)
#         self.total_distance += distances[c_city][next_city]
    
#     def update_edges_pheromone(self):
#         # Deposit pheromone on the edges of the ant's tour
#         for i in range(len(self.v_cities) - 1):
#             c_city = self.v_cities[i]
#             next_city = self.v_cities[i+1]
#             pheromone[c_city][next_city] += 1.0 / self.total_distance

# ants = [Ant(0) for i in range(num_ants)]

# for iteration in range(num_iterations):
#     for ant in ants:
#         while len(ant.v_cities) < len(distances):
#             ant.select_next_city()
#         ant.total_distance += distances[ant.v_cities[-1]][ant.v_cities[0]]
    
#     for ant in ants:
#         ant.update_edges_pheromone()
    
#     # # Evaporate the pheromone on all edges
#     # for i in range(len(distances)):
#     #     for j in range(len(distances)):
#     #         pheromone[i][j] *= (1.0 - evaporation)
    
#     # Print the best solution found so far
#     best_ant = min(ants, key=lambda x: x.total_distance)
#     print("Iteration {}: Best Distance = {}".format(iteration, best_ant.total_distance))