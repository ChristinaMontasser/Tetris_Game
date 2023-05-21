import random

import random
import math


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
                distance =int(math.sqrt((xi - xj)**2 + (yi - yj)**2))
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
            next_city = random.choices([c for c, p in probs])[0]
            # weights = [p / total_pheromone for c, pin probs]
            # next_city = random.choices([c for c, p in probs], weights=weights)[0]

        self.v_cities.append(next_city)
        self.total_distance += self.distances[c_city][next_city]
        # print(self.total_distance)

    def update_edges_pheromone(self, pheromone):
        #Emphasis on the best route  
        for i in range(len(self.v_cities) - 1):
            c_city = self.v_cities[i]
            next_city = self.v_cities[i+1]
            pheromone[c_city][next_city] += 1.0 / self.total_distance
        #Evaporating 
        for i in range(len(pheromone)):
            for j in range(len(pheromone[i])):
                pheromone[i][j] *= (1.0 - evaporation)
              
        return pheromone

    def reset_ant(self):
        self.v_cities = [self.start_city]
        self.total_distance = 0.0

def run(num_of_ants, num_iterations, distances, pheromone, cities):
    # start_cities= [random.randint(0,len(cities)-1) for i in range(num_ants)]
    # ants = [Ant(start_cities[i], distances, cities) for i in range(num_ants)]
    
    start_city = 0
    ants = [Ant(start_city, distances, cities) for i in range(num_of_ants)]
    for iteration in range(num_iterations):
        pheromone_map=[]
        for ant in ants:
            while len(ant.v_cities) < len(distances):
                ant.select_next_city(pheromone)
            ant.total_distance += distances[ant.v_cities[-1]][ant.v_cities[0]]  # Return back to the first city
            ant.v_cities.append(start_city)

        for ant in ants:
            pheromone = ant.update_edges_pheromone(pheromone)

        if(iteration%10==0):
            for ant in ants:   
              pheromone_ant=[]
              for i in range(len(ant.v_cities) - 1):
                    c_city = ant.v_cities[i]
                    next_city = ant.v_cities[i+1]
                    pheromone_ant.append(pheromone[c_city][next_city])
              pheromone_map.append(pheromone_ant)
            best_ant_index, best_ant= min(enumerate(ants), key=lambda x: x[1].total_distance)
            print("Iteration {}:".format(iteration))
            print("Best Distance = {}".format(best_ant.total_distance))
            print("Pheromone Map = {}".format(pheromone_map))
            print("Pheromone Best Ant = {} \n".format(pheromone_map[best_ant_index]))
            print("Visited {}".format(best_ant.v_cities))


        for ant in ants:
            #print("Total distance {}".format(ant.total_distance))
            ant.reset_ant()

if __name__ == "__main__":
    
    num_ants = 10
    alpha = 1.0
    beta = 2.0
    evaporation = 0.1
    exploration = 0.9
    num_iterations = 100

    distances1, pheromone1, cities1 = generate_cities(num_cities=10, min_distance=3, max_distance=40)
    distances1