import random

import numpy as np
from random import Random


def condensed_print(matrix):
    for i in matrix:
        for j in i:
            print(j, end='')
        print()


def print_all_forms():
    for piece in TetrisEnv.Pieces:
        print(piece + ":")
        print('---')
        condensed_print(TetrisEnv.Pieces[piece])
        print('#')
        condensed_print(np.rot90(TetrisEnv.Pieces[piece], axes=(1, 0)))
        print('#')
        condensed_print(np.rot90(TetrisEnv.Pieces[piece], 2, axes=(1, 0)))
        print('#')
        condensed_print(np.rot90(TetrisEnv.Pieces[piece], 3, axes=(1, 0)))
        print('---')
        print()


class TetrisEnv:
    SCORE_PIXEL = 1
    SCORE_SINGLE = 40 * 10
    SCORE_DOUBLE = 100 * 10
    SCORE_TRIPLE = 300 * 10
    SCORE_TETRIS = 1200 * 10
    MAX_TETRIS_ROWS = 20
    GAMEOVER_ROWS = 4
    TOTAL_ROWS = MAX_TETRIS_ROWS + GAMEOVER_ROWS
    MAX_TETRIS_COLS = 10
    GAMEOVER_PENALTY = -1000
    TETRIS_GRID = (TOTAL_ROWS, MAX_TETRIS_COLS)
    TETRIS_PIECES = ['O', 'I', 'S', 'Z', 'T', 'L', 'J']
    # Note, pieces are rotated clockwise
    Pieces = {'O': np.ones((2, 2), dtype=np.byte),
              'I': np.ones((4, 1), dtype=np.byte),
              'S': np.array([[0, 1, 1], 
                             [1, 1, 0]], dtype=np.byte),
              'Z': np.array([[1, 1, 0], 
                             [0, 1, 1]], dtype=np.byte),
              'T': np.array([[1, 1, 1], 
                             [0, 1, 0]], dtype=np.byte),
              'L': np.array([[1, 0], 
                             [1, 0], 
                             [1, 1]], dtype=np.byte),
              'J': np.array([[0, 1], 
                             [0, 1], [1, 1]], dtype=np.byte),
              }
    '''
    I:   S:      Z:      T:
      1      1 1    1 1     1 1 1
      1    1 1        1 1     1
      1
      1
    L:      J:      O:
      1        1      1 1
      1        1      1 1
      1 1    1 1
     last one is utf
    '''

    def __init__(self):
        self.RNG = Random()  # independent RNG
        self.default_seed = 17  # default seed is IT
        self.__restart()



    def __restart(self):
        self.RNG.seed(self.default_seed)
        self.board = np.zeros(self.TETRIS_GRID, dtype=np.byte)
        self.current_piece = self.RNG.choice(self.TETRIS_PIECES)
        self.next_piece = self.RNG.choice(self.TETRIS_PIECES)
        self.score = 0

    def __gen_next_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self.RNG.choice(self.TETRIS_PIECES)

    def set_seed(self, seed_value):
        self.default_seed = seed_value

    def get_status(self):
        return self.board.copy(), self.current_piece, self.next_piece

    # while can move down piece, move it down (note to restrict col to rotation max)
    # which is COLS-1 - (piece width in cur rotation -1) or cancel both -1s utf-8 #
    # check if move down, row++, if not, print piece on last row, col
    def __get_score(self, value):
        if value == 1:
            return TetrisEnv.SCORE_SINGLE
        if value == 2:
            return TetrisEnv.SCORE_DOUBLE
        if value == 3:
            return TetrisEnv.SCORE_TRIPLE
        if value == 4:
            return TetrisEnv.SCORE_TETRIS
        return 0
        
    def __collapse_rows(self, board):
        start_collapse = -1
        for row, i in zip(board, range(TetrisEnv.TOTAL_ROWS)):
            if np.sum(row) == TetrisEnv.MAX_TETRIS_COLS:
                start_collapse = i
                break
        if start_collapse == -1:
            return 0, board
        end_collapse = start_collapse + 1
        while end_collapse < TetrisEnv.TOTAL_ROWS:
            if np.sum(board[end_collapse]) == TetrisEnv.MAX_TETRIS_COLS:
                end_collapse += 1
            else:
                break
        new_board = np.delete(board, slice(start_collapse, end_collapse), axis=0)  # now we need to add them
        new_board = np.insert(new_board, slice(0, end_collapse - start_collapse), 0, axis=0)
        score = self.__get_score(end_collapse - start_collapse)

        return score, new_board

    def __game_over(self, test_board):
        return np.sum(test_board[:TetrisEnv.GAMEOVER_ROWS]) > 0

    def __play(self, col, rot_count):
        falling_piece = self.Pieces[self.current_piece]
        if rot_count > 0:
            falling_piece = np.rot90(falling_piece, rot_count, axes=(1, 0))
        p_dims = falling_piece.shape
        col = min(col, TetrisEnv.MAX_TETRIS_COLS - p_dims[1])
        max_row = TetrisEnv.TOTAL_ROWS - p_dims[0]
        chosen_row = 0
        while chosen_row < max_row:
            next_row = chosen_row + 1
            if np.sum(np.multiply(falling_piece,
                    self.board[next_row:next_row + p_dims[0], col:col + p_dims[1]])) > 0:
                break
            chosen_row = next_row
        self.board[chosen_row:chosen_row + p_dims[0], col:col + p_dims[1]] |= falling_piece
        collapse_score, new_board = self.__collapse_rows(self.board)
        collapse_score += np.sum(falling_piece) * TetrisEnv.SCORE_PIXEL
        if self.__game_over(new_board):
            return TetrisEnv.GAMEOVER_PENALTY
        self.board = new_board
        return collapse_score

    # does not affect the class, tests a play of the game given a board and a piece b64 #
    def test_play(self, board_copy, piece_type, col, rot_count):
        falling_piece = self.Pieces[piece_type]
        if rot_count > 0:
            falling_piece = np.rot90(falling_piece, rot_count, axes=(1, 0))
        p_dims = falling_piece.shape
        col = min(col, TetrisEnv.MAX_TETRIS_COLS - p_dims[1])
        max_row = TetrisEnv.TOTAL_ROWS - p_dims[0]
        chosen_row = 0
        while chosen_row < max_row:
            next_row = chosen_row + 1
            if np.sum(np.multiply(falling_piece,
                                  board_copy[next_row:next_row + p_dims[0], col:col + p_dims[1]])) > 0:
                break
            chosen_row = next_row
        board_copy[chosen_row:chosen_row + p_dims[0], col:col + p_dims[1]] |= falling_piece
        collapse_score, board_copy = self.__collapse_rows(board_copy)
        collapse_score += np.sum(falling_piece) * TetrisEnv.SCORE_PIXEL
        if self.__game_over(board_copy):
            return TetrisEnv.GAMEOVER_PENALTY, board_copy
        return collapse_score, board_copy

    def __calc_rank_n_rot(self, scoring_function, genetic_params, col):
        # should return rank score and rotation a pair (rank,rot), rot is from 0 to 3
        return scoring_function(self, genetic_params, col)

    def __get_lose_msg(self):
        # if understood, send to owner
        lose_msg = b'Hello'
        return lose_msg

    def run(self, scoring_function, genetic_params, num_of_iters, return_trace):
        self.__restart()
        # no trace
        if not return_trace:
            for it in range(num_of_iters):
                rates = []
                rotations = []
                for c in range(TetrisEnv.MAX_TETRIS_COLS):
                    r1, r2 = self.__calc_rank_n_rot(scoring_function, genetic_params, c)
                    rates.append(r1)
                    rotations.append(r2)
                pos_to_play = rates.index(max(rates))  # plays first max found
                rot_to_play = rotations[pos_to_play]
                play_score = self.__play(pos_to_play, rot_to_play)
                self.score += play_score
                self.__gen_next_piece()
                if play_score < 0:
                    return self.score, self.board, self.__get_lose_msg()
            return self.score, self.board, ""
        else:  # we want to trace
            board_states = []
            ratings_n_rotations = []
            pieces_got = []
            # board_states.append(self.board.copy())
            for it in range(num_of_iters):
                rates = []
                rotations = []
                pieces_got.append(self.current_piece)
                for c in range(TetrisEnv.MAX_TETRIS_COLS):
                    r1, r2 = self.__calc_rank_n_rot(scoring_function, genetic_params, c)
                    rates.append(r1)
                    rotations.append(r2)
                ratings_n_rotations.append(list(zip(rates, rotations)))
                pos_to_play = rates.index(max(rates))  # plays first max found
                rot_to_play = rotations[pos_to_play]
                play_score = self.__play(pos_to_play, rot_to_play)
                self.score += play_score
                self.__gen_next_piece()
                board_states.append(self.board.copy())
                if play_score < 0:
                    return self.score, board_states, ratings_n_rotations, pieces_got, self.__get_lose_msg()
            return self.score, board_states, ratings_n_rotations, pieces_got, ""
        # don't really feel like removing redundancy, cleaning code

def number_of_holes(board):
    height = len(board)
    width = len(board[0])
    gaps = 0
    for col in range(width-1):
        gap = False
        for row in range(height-1, 0, -1):
            if not board[row][col]:
              if ((board[row+1][col] or gap) and (board[row][col-1] or col-1 == 0 or board[row-1][col-1]) and (board[row][col+1] or col == width-1 or board[row+1][col+1])):
                    gaps += 1
                    gap = True
              else: 
                  gap = False
            else: 
                  gap = False
    return gaps

def count_complete_rows(board):
    height = len(board)
    complete_rows = 0
    for row in range(height):
        if all(board[row]):
            complete_rows += 1
    return complete_rows

def calculate_bumpiness(board):
    heights = [0] * len(board[0])
    for col in range(len(board[0])):
        for row in range(len(board)):
            if board[row][col] == 1:
                heights[col] = len(board) - row
                break
    bumpiness = 0
    for i in range(len(heights) - 1): #absoluate difference 
        bumpiness += abs(heights[i] - heights[i+1])
    return bumpiness

def calculate_landing_height(board, block, column, i):
    falling_piece = np.rot90(block, i, axes=(1, 0))
    falling_piece_height = len(falling_piece)
    for row in range(len(board) - 1, -1, -1): 
        #20
        #That first condition guarantee that we don't go deeper into the column until we reach the first row
        #Because there's no piece whose height is only 1, so the logic is conditioned by the piece height itself (Empty column)
        if row - falling_piece_height + 1 < 0 or \
           any(board[row - i][column] for i in range(falling_piece_height)):
            #It checks if any of the cells in the specified column and 
            # the rows (with heighest block) below the current row are already occupied by existing blocks.
            return row + 1
    return 0

# max gain + random
def random_scoring_function(tetris_env: TetrisEnv, gen_params, col):
    #Number of holes (after - before)
    #Bumpiness
    #Complete rows (after - before) -->clear rows
    #Number of block transitions:
    #fitness = (number of cleared rows) + (sum of heights of columns) - (landing height penalty)
    #Others:
        #Aggregate height
        #Number of holes
    #     #Blockades
    board, piece, next_piece = tetris_env.get_status()  # add type hinting
    rows_before=count_complete_rows(board)
    val1_scores = []
    val2_bumpiness = []
    val3_landing_height = []
    val4_clear_rows_after = []
    for i in range(4):
        score, tmp_board = tetris_env.test_play(board, piece, col, i)
        val4_clear_rows_after.append(count_complete_rows(tmp_board)-rows_before)
        val2_bumpiness.append(calculate_bumpiness(tmp_board))
        l_piece=tetris_env.Pieces[piece]
        val3_landing_height.append(calculate_landing_height(board, l_piece, col, i))

        if score < 0:
            val1_scores.append(score )
            continue
        tmp_val1_scores = []
        for t in range(tetris_env.MAX_TETRIS_COLS):
            for j in range(4):
                score2, _ = tetris_env.test_play(tmp_board, next_piece, t, j)
                tmp_val1_scores.append(score2 )
        max_score2 = max(tmp_val1_scores)
        if max_score2 >= 0:
            score += max_score2
        val1_scores.append(score)

    fitness = [gen_params[0]*val1_scores[i] + gen_params[1]*val2_bumpiness[i] + gen_params[2]*val3_landing_height[i] + gen_params[3]*val4_clear_rows_after[i] for i in range(4)]
    index, rate = max(enumerate(fitness), key=lambda item: item[1])  # need to store it first or it iterates
    # print(val)
    return  rate, index


def print_stats(use_visuals_in_trace_p, states_p, pieces_p, sleep_time_p):
    vision = BoardVision()
    if use_visuals_in_trace_p:

        for state, piece in zip(states_p, pieces_p):
            vision.update_board(state)
            time.sleep(sleep_time_p)
        time.sleep(2)
        vision.close()
    else:
        for state, piece in zip(states_p, pieces_p):
            print("board")
            condensed_print(state)
            print("piece")
            condensed_print(piece)
            print('-----')


class GA:
    def __init__(self, pop_size= 10):
        self.init_pop_size= pop_size
        self.population = []
        self.init_population()

    def init_population(self):
        for i in range(self.init_pop_size):
            chromosome = [random.uniform(1, 10), random.uniform(-6, -1), random.uniform(1, 5), random.uniform(1, 5)]
            self.population.append(chromosome)
        
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
        child_2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child_1, child_2
    

    def __mutation_operator(self, chromosome):
        mutation_rate = 0.05
        mutated_chromosome = []
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                new_gene = random.uniform(-10, 10)
                mutated_chromosome.append(new_gene)
            else:
                mutated_chromosome.append(chromosome[i])
        return mutated_chromosome
    
    
    def __get_best_individual(self):
        # Return the best individual (chromosome)
        best_index = np.argmax(self.fitness_scores)
        return best_index

    def evolve_population(self, fitness_scores):
        self.fitness_scores =  fitness_scores
        elite_size= len(self.population)-1
        best_individual_indices = sorted(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i], reverse=True)[:elite_size]
        #best_individual_values = [self.population[i] for i in best_individual_indices]
        # print("best_individual_indices {}".format(best_individual_indices))
        parents_indeces = self.__selection_operator(best_individual_indices, len(best_individual_indices)-1)

        new_population = []
        # print("self.population {}".format(self.population))
        # print("parents_indeces {}".format(len(parents_indeces)))
        # print("parents_indeces {}".format(parents_indeces))
        for i in range(int((len(self.population)/2)-1)):
            parent1_index, parent2_index = random.sample(parents_indeces, 2)
            # print("parent1_index {}".format(parent1_index))
            # print("parent2_index {}".format(parent2_index))
            child_1, child_2 = self.__crossover_operator(self.population[parent1_index], self.population[parent2_index])
            child_1 = self.__mutation_operator(child_1)
            child_2 = self.__mutation_operator(child_2)
            new_population.append(child_1)
            new_population.append(child_2)

        self.population = new_population 
        print(self.__get_best_individual())
        return self.population
        



if __name__ == "__main__":
    use_visuals_in_trace = False
    sleep_time = 0.8
    pop_size= 20
    eposide = 10 
    # just one chromosome in the population
    ga_algo=GA(pop_size)
    population = ga_algo.population
    #chromo_rando = [[3, -2, -2, 4], [3, -2, 2, 4]] #Send the best one
    # population = [[2.8673550667, -4.915090563907668, 1.7326229294816473, 4.303487143315175], [2.8673550667, -4, 1.7326229294816473, 4.303487143315175],
    #                [3, -4.915090563907668, 2, 5], [2.8673550667, -4.915090563907668, 4, 4.303487143315175],
    #                [4, -4.915090563907668, 1.7326229294816473, 4.303487143315175]]
    #one_chromo_competent = [-4, -1, 2,3]
    from Visor import BoardVision
    import time
    
    env = TetrisEnv()
    seed = 5132
    env.set_seed(seed)
    total_scores = []
    with open('output.txt', 'a') as f:
            f.write("Exprience 3 \n")

    for k in range (eposide):
        for i in range(len(population)):
            total_score, states, rate_rot, pieces, msg = env.run(
                random_scoring_function, population[i], 100, True)
            # after running your iterations (which should be at least 500 for each chromosome)
            # you can evolve your new chromosomes from the best after you test all chromosomes here
            total_scores.append(total_score)
            # print("Ratings and rotations")
            # for rr in rate_rot:
            #     print(rr)
            print('----')
            print(total_score)
            # print(msg)
            #print_stats(use_visuals_in_trace, states, pieces, sleep_time)

        with open('output.txt', 'a') as f:
            f.write("Iteration {} \n".format(k))
            f.write("Cofig: Seed {} \n".format(seed))
            f.write("Population is: {} \n".format(population))
            f.write("total_score is: {} \n".format(total_scores))
            
        population = ga_algo.evolve_population(total_scores)
        if(len(population)<=1):
            break
        total_scores = []
