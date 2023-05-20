import random

import random
import math

num_ants = 10
alpha = 1.0
beta = 2.0
evaporation = 0.1
exploration = 0.9
num_iterations = 100

def generate_cities(num_cities=5, pheromone_initial=1, min_distance= 0, max_distance=100):
    cities = []
    for i in range(num_cities):
        x = random.randint(min_distance, max_distance)
        y = random.randint(min_distance, max_distance)
        cities.append((x, y))
     
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




class Ant:
    def __init__(self, start_city, distances, cities):
        self.start_city = start_city
        self.v_cities = [start_city]
        self.total_distance = 0.0
        self.distances = distances
        self.cities =cities

    def select_next_city(self, pheromone):
        probs = []
        c_city = self.v_cities[-1] 
          #select current city by selecting the last city of visited
        total_pheromone = 0.0
        for city in range(len(self.distances)):
            if city not in self.v_cities:
                pheromone_level = pheromone[c_city][city]
                distance = self.distances[c_city][city]
                probability = (pheromone_level ** alpha) * ((1.0 / distance) ** beta)
                probs.append((city, probability))
                total_pheromone += probability
                #(T^alpha*(1/d)^beta)/(Sum(T^alpha*(1/d)^beta))

        if random.random() < exploration: # (exploitation)
            probs.sort(key=lambda x: x[1], reverse=True)
            next_city = probs[0][0]
        else: # (exploration)
            probs = [(c, p / total_pheromone) for c, p in probs]
            next_city = random.choices([c for c, p in probs], weights=[p for c, p in probs])[0]
        # print(next_city)
            #print(next_city)
        self.v_cities.append(next_city)
        self.total_distance += self.distances[c_city][next_city]
        # print(self.total_distance)

    def update_edges_pheromone(self, pheromone):
        #Emphasis on the best route  
        for i in range(len(self.v_cities) - 1):
            c_city = self.v_cities[i]
            next_city = self.v_cities[i+1]
            pheromone[c_city][next_city] += 2.0 / self.total_distance
            # print("Here")
        #Evaporating 
        for i in range(len(pheromone)):
            for j in range(len(pheromone[i])):
                pheromone[i][j] *= (1.0 - evaporation)
              
        return pheromone

    def reset_ant(self):
        self.v_cities = [self.start_city]
        self.total_distance = 0.0



distances, pheromone, cities =generate_cities(num_cities=20, min_distance=1, max_distance=50)

#Let each ant starts from a different city 
# start_cities= [random.randint(0,19) for i in range(num_ants)]
# ants = [Ant(start_cities[i], distances, cities) for i in range(num_ants)]

#All start from city 0
ants = [Ant(0, distances, cities) for i in range(num_ants)]

#print(start_cities)
for iteration in range(20):
  for ant in ants:
      while len(ant.v_cities) < len(distances):
          ant.select_next_city(pheromone)
      ant.total_distance += distances[ant.v_cities[-1]][ant.v_cities[0]] #Return back to the first city 

  for ant in ants:
      pheromone = ant.update_edges_pheromone(pheromone)

  best_ant = min(ants, key=lambda x: x.total_distance)
  print("Iteration {}: Best Distance = {}".format(iteration, best_ant.total_distance))

  for ant in ants:
    #print(ant.total_distance)
    ant.reset_ant()
