import random

import numpy as np
from random import Random

def number_of_holes(board):
    height = len(board)
    width = len(board[0])
    gaps = 0
    for col in range(width-1):
        gap = False
        for row in range(height-2, 0, -1):
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

def count_block_transitions(board):
    height = len(board)
    width = len(board[0])
    transitions = 0

    for row in range(height - 1):
        for col in range(width):
            if board[row][col] != board[row + 1][col]:
                transitions += 1

    for col in range(width - 1):
        for row in range(height):
            if board[row][col] != board[row][col + 1]:
                transitions += 1

    return transitions


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