import numpy as np
import random

class GA:
    def __init__(self, pop_size, num_factors=4, mutation_rate=0.03):
        self.pop_size = pop_size
        self.num_factors = num_factors
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        population = []
        for i in range(self.pop_size):
            chromosome = np.random.rand(self.num_factors)
            population.append(chromosome)
        return population

    def fitness_function(self, board, piece, move, factors):
        # Create a copy of the board and apply the move to it
        test_board = board.copy()
        test_board.apply_move(piece, move)

        # Calculate the number of holes in the board
        holes = 0
        for col in range(test_board.width):
            # Find the first occupied cell in the column
            row = 0
            while row < test_board.height and test_board.get_cell(row, col) == 0:
                row += 1

            # Count the number of empty cells below the first occupied cell
            for r in range(row+1, test_board.height):
                if test_board.get_cell(r, col) == 0:
                    holes += 1

        # Calculate the height of the highest column
        heights = [test_board.height - row for row in test_board.get_heights()]
        max_height = max(heights)

        # Calculate the total height of the columns
        total_height = sum(heights)

        # Calculate the bumpiness of the columns
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(test_board.width-1))

        # Calculate the fitness score
        fitness = factors[0] * total_height + factors[1] * holes + factors[2] * bumpiness + factors[3] * max_height + factors[4]

        return fitness
    def selection_operator(self, fitness_scores):
        idx1 = random.randint(0, len(fitness_scores) - 1)
        idx2 = random.randint(0, len(fitness_scores) - 1)
        if fitness_scores[idx1] > fitness_scores[idx2]:
            return idx1
        else:
            return idx2

    def crossover_operator(self, parent1, parent2):
        # Use uniform crossover to combine the chromosomes of the parents
        # Choose a random crossover point for each chromosome
        child1 = []
        child2 = []
        for i in range(len(parent1)):
            if random.random() < self.crossover_rate:
                child1.append(parent2[i])
                child2.append(parent1[i])
            else:
                child1.append(parent1[i])
                child2.append(parent2[i])
        return child1, child2
    
    def mutation_operator(self, chromosome):
        # Introduce random changes to the chromosomes of the individuals
        # Flip the bits of each chromosome with probability mutation_rate
        mutated_chromosome = []
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                mutated_chromosome.append(1 - chromosome[i])
            else:
                mutated_chromosome.append(chromosome[i])
        return mutated_chromosome

    def evolve_population(self, board, piece, move):
        # Evaluate the fitness of each individual in the population
        fitness_scores = []
        for chromosome in self.population:
            fitness_score = self.fitness_function(board, piece, move, chromosome)
            fitness_scores.append(fitness_score)

        # Select the fittest individuals to be the parents of the next generation
        parents = self.selection_operator(fitness_scores)

        # Use the parents to create new individuals through crossover and mutation
        new_population = []
        for i in range(self.pop_size):
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover_operator(parent1, parent2)
            child = self.mutation_operator(child)
            new_population.append(child)

        # Update the population
        self.population = new_population

    def get_best_individual(self, board, piece, move):
        # Evaluate the fitness of each individual in the population
        fitness_scores = []
        for chromosome in self.population:
            fitness_score = self.fitness_function(board, piece, move, chromosome)
            fitness_scores.append(fitness_score)

        # Return the best individual (chromosome)
        best_index = np.argmax(fitness_scores)
        return self.population[best_index]
    
ga = GA()
ga.