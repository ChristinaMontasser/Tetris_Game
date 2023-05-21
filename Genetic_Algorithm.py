import random

import numpy as np
from random import Random


class GA:
    def __init__(self, pop_size= 10):
        self.init_pop_size= pop_size
        self.population = []
        self.init_population()

    def init_population(self):
        for i in range(self.init_pop_size):
            chromosome = [random.uniform(1, 10), random.uniform(-6, -1), random.uniform(-4, -1), 
                          random.uniform(1, 4), random.uniform(-3, 0), random.uniform(-4, -1) ]
            self.population.append(chromosome)
    #val1_scores = []
    # val2_bumpiness = []
    # val3_landing_height = []
    # val4_clear_rows_after = []
    # val5_number_of_holes = []
    # val6_block_transitions = []
    # val7_second_scores = []
        
    def __selection_operator(self, best_individual_indices: list,  number): # len(fitness_scores) = number of(choromsss)
        candidates = []
        for i in range(number):
            idx1 = random.choice(best_individual_indices)
            idx2 = random.choice(best_individual_indices)
            if self.fitness_scores[idx1] > self.fitness_scores[idx2]:
                candidates.append(idx1)
                best_individual_indices.pop(best_individual_indices.index(idx1))
            else:
                candidates.append(idx2)
                best_individual_indices.pop(best_individual_indices.index(idx2))
        candidates
        return candidates


    def __crossover_operator(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child_1 = parent1[:crossover_point] + parent2[crossover_point:]
        return child_1
    

    def __mutation_operator(self, chromosome):
        mutation_rate = 0.05
        mutated_chromosome = []
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                new_gene = random.uniform(-2,2)
                mutated_chromosome.append(chromosome[i]+new_gene)
            else:
                mutated_chromosome.append(chromosome[i])
        return mutated_chromosome
    
    def __get_best_individual(self):
        # Return the best individual (chromosome)
        best_index = np.argmax(self.fitness_scores)
        return best_index

    def evolve_population(self, fitness_scores, its):
        self.fitness_scores =  fitness_scores
        elite_size= int(len(self.population)*0.75)
        best_individual_indices = sorted(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i], reverse=True)[:elite_size]
        #best_individual_values = [self.population[i] for i in best_individual_indices]
        # print("best_individual_indices {}".format(best_individual_indices))
        parents_indeces = self.__selection_operator(best_individual_indices, len(best_individual_indices)-1)

        new_population = []
        for i in range(int((len(best_individual_indices)/2)-2)):
            parent1_index, parent2_index = random.sample(parents_indeces, 2)
            # print("parent1_index {}".format(parent1_index))
            # print("parent2_index {}".format(parent2_index))
            child_1 = self.__crossover_operator(self.population[parent1_index], self.population[parent2_index])
            child_1 = self.__mutation_operator(child_1)
            #child_2 = self.__mutation_operator(child_2)
            new_population.append(child_1)
            #new_population.append(child_2)
        for i in best_individual_indices[:int(len(best_individual_indices)/4)]:
            new_population.append(self.population[i])
        self.population = new_population 
        print(self.__get_best_individual())
        return self.population
        

