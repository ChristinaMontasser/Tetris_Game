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
     #   self.population = self.init_population()

    # def init_population(self):
    #     population = []
    #     for i in range(self.pop_size):
    #         chromosome = np.random.rand(self.num_factors)
    #         population.append(chromosome)
    #     return population

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

# max gain + random
def random_scoring_function(tetris_env: TetrisEnv, gen_params, col):
    board, piece, next_piece = tetris_env.get_status()  # add type hinting
    scores = []
    for i in range(4):
        score, tmp_board = tetris_env.test_play(board, piece, col, i)
        if score < 0:
            scores.append([score * gen_params[0], i])
            continue
        tmp_scores = []
        for t in range(tetris_env.MAX_TETRIS_COLS):
            for j in range(4):
                score2, _ = tetris_env.test_play(tmp_board, next_piece, t, j)
                tmp_scores.append(score2 * gen_params[1])
        max_score2 = max(tmp_scores)
        if max_score2 >= 0:
            score += max_score2
        scores.append([score * gen_params[0], i])
    for i in range(4):
        scores[i][0] *= random.randint(1, gen_params[2])
    val = max(scores, key=lambda item: item[0])  # need to store it first or it iterates
    print(val)
    return val[0], val[1]


def print_stats(use_visuals_in_trace_p, states_p, pieces_p, sleep_time_p):
    vision = BoardVision()
    if use_visuals_in_trace_p:

        for state, piece in zip(states_p, pieces_p):
            vision.update_board(state)
            # print("piece")
            # condensed_print(piece)
            # print('-----')
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

# def selection_operator(self, fitness_scores):
#         idx1 = random.randint(0, len(fitness_scores) - 1)
#         idx2 = random.randint(0, len(fitness_scores) - 1)
#         if fitness_scores[idx1] > fitness_scores[idx2]:
#             return idx1
#         else:
#             return idx2

# def crossover_operator(self, parent1, parent2):
#     # Use uniform crossover to combine the chromosomes of the parents
#     # Choose a random crossover point for each chromosome
#     child1 = []
#     child2 = []
#     for i in range(len(parent1)):
#         if random.random() < self.crossover_rate:
#             child1.append(parent2[i])
#             child2.append(parent1[i])
#         else:
#             child1.append(parent1[i])
#             child2.append(parent2[i])
#     return child1, child2

# def mutation_operator(self, chromosome):
#     # Introduce random changes to the chromosomes of the individuals
#     # Flip the bits of each chromosome with probability mutation_rate
#     mutated_chromosome = []
#     for i in range(len(chromosome)):
#         if random.random() < self.mutation_rate:
#             mutated_chromosome.append(1 - chromosome[i])
#         else:
#             mutated_chromosome.append(chromosome[i])
#     return mutated_chromosome

# def evolve_population(self, board, piece, move):
#     # Evaluate the fitness of each individual in the population
#     fitness_scores = []
#     for chromosome in self.population:
#         fitness_score = self.fitness_function(board, piece, move, chromosome)
#         fitness_scores.append(fitness_score)

#     # Select the fittest individuals to be the parents of the next generation
#     parents = self.selection_operator(fitness_scores)

#     # Use the parents to create new individuals through crossover and mutation
#     new_population = []
#     for i in range(self.pop_size):
#         parent1, parent2 = random.sample(parents, 2)
#         child = self.crossover_operator(parent1, parent2)
#         child = self.mutation_operator(child)
#         new_population.append(child)

#     # Update the population
#     self.population = new_population

# def get_best_individual(self, board, piece, move):
#     # Evaluate the fitness of each individual in the population
#     fitness_scores = []
#     for chromosome in self.population:
#         fitness_score = self.fitness_function(board, piece, move, chromosome)
#         fitness_scores.append(fitness_score)

#     # Return the best individual (chromosome)
#     best_index = np.argmax(fitness_scores)
#     return self.population[best_index]

if __name__ == "__main__":
    use_visuals_in_trace = True
    sleep_time = 0.8
    # just one chromosome in the population
    one_chromo_rando = [[1, 1, 4], [2, 1, 5]] #Send the best one
    #one_chromo_competent = [-4, -1, 2,3]
    from Visor import BoardVision
    import time

    # print_all_forms()
    env = TetrisEnv()
    env.set_seed(5132)
    for i in range(len(one_chromo_rando)):
        total_score, states, rate_rot, pieces, msg = env.run(
            random_scoring_function, one_chromo_rando[i], 100, True)
        # after running your iterations (which should be at least 500 for each chromosome)
        # you can evolve your new chromosomes from the best after you test all chromosomes here
        print("Ratings and rotations")
        print(len(rate_rot[2]))
        for rr in rate_rot:
            print(rr)
        print('----')
        print('I am chromosome: {}'.format(i+1))
        print(total_score)
        print(msg)
        print_stats(use_visuals_in_trace, states, pieces, sleep_time)
        
