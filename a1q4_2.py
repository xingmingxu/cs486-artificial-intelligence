#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:37:32 2023

@author: xingmingxu
"""

import numpy as np
import random
import time

N = 9
count_1 = 0

class Sudoku:
    
    """
    Members:
        - grid: Sudoku grid with cells: (0, n^2-1)
        - assignment: Possible values at the i'th position in grid.
    """
    
    def __init__(self, input_string):
        self.grid = np.zeros((N, N))
        spaceless = input_string.replace('\n','')
        spaceless = spaceless.replace(' ','')
        #sprint(spaceless)
        for i in range(N):
            for j in range(N):
                self.grid[i][j] = spaceless[i * N + j]
              
    def solve(self, option: str):
        if option == "A":
            return backtrack(self.grid)
        if option == "B":
            
            #assignments = create_assignments(self.grid)
            #for i in range(N):
                 #for j in range(N):
                     #print(self.grid[i][j], assignments[(i,j)])
            #assignments = update_assignments(self.grid, assignments)
            #print("updated assignments")
            #for i in range(N):
            #     for j in range(N):
            #         print((i,j),self.grid[i][j], assignments[(i,j)])
            print(self.grid)
            
            return backtrack_forwardcheck(self.grid)
        
        if option == "C":
            return backtrack_fcheck_heuristics(self.grid)
        
       
# Check constraints
def check_row(val: int, row: int, col: int, grid) -> bool:

    """
    Checks if inserting val meets the constraints of the row of loc
    """
    #row = loc // N
    #col = loc % N
    for c in range(N):
       # if c == col:
           # continue
       if grid[row][c] == val:
           return False
    return True


def check_col(val: int, row: int, col: int, grid) -> bool:
    """
    Checks if inserting val meets the constraints of the col of loc
    """
    for r in range(N):
       if grid[r][col] == val:
           return False
    return True


def check_set(val: int, row: int, col: int, grid) -> bool:
    """
    Checks constraints for 3x3 grid.
    """
    start_row = (int)(row - row % 3)
    start_col = (int)(col - col % 3)
    for i in range(3):
        for j in range(3):
            if start_row + i == row and start_col + j == col:
                continue
            if grid[start_row + i][start_col + j] == val:
                return False
    return True
    

def check_val(val: int, row: int, col: int, grid) -> bool:
    
    """
    Checks if a value being placed at (row, col) in grid
    violates any constraints.
    """
    crow = check_row(val, row, col, grid)
    ccol = check_col(val, row, col, grid)
    cset = check_set(val, row, col, grid)
    return crow and ccol and cset


def get_unassigned_variables(grid) -> list:
    
    """
    Retrieves list of empty cells
    """
    retlist = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 0:
                retlist.append((i,j))
    return retlist


def update_constraints(grid, assignments, row, col) -> dict:
    
    """
    Updates the constraints for grid
    """
    val = grid[row][col]
    for c in range(N):
        assignments[(row, c)].discard(val)
    for r in range(N):
        assignments[(r, col)].discard(val)
        
    start_row = (int)(row - row % 3)
    start_col = (int)(col - col % 3)
    for i in range(3):
        for j in range(3):
            assignments[(start_row + i, start_col + j)].discard(val)
    return assignments

    
def create_assignments(grid) -> dict:
    
    """
    Gets list of assignments for a blank grid.
    """
    assignments = dict()
    for i in range(N):
        for j in range(N):
            assignments[(i,j)] = set([1,2,3,4,5,6,7,8,9])

    for r in range(N):
        for c in range(N):
            if grid[r][c] != 0:
                update_constraints(grid, assignments, r, c)
    return assignments


# OPTION A      
def backtrack(grid):
    
    if 0 not in grid:
        return grid
    
    # select a random unassigned variable: this is an assignment key.
    unassigned_variables = get_unassigned_variables(grid)
    if len(unassigned_variables) == 0:
        return None
    var = random.choice(unassigned_variables)

    r = var[0]
    c = var[1]
    
    domain = [x for x in range(1,10)]
    
    while len(domain) != 0:
        value = random.choice(domain)
        domain.remove(value)
        if check_val(value, r, c, grid):
            grid[r][c] = value
            result = backtrack(grid)
            if result is not None and 0 not in result:
                return result
            grid[r][c] = 0
    
    return None


# OPTION B
def forwardcheck(grid, assignments, row, col, val) -> bool:
    
    """
    Check if the value we want to insert would reduce anything
    to a domain of 0.
    """
    for c in range(N):
        if c == col or grid[row][c] != 0:
            continue
        domain = assignments[(row, c)]
        if len(domain) == 1 and val in domain:
            return False
        
    for r in range(N):
        if r == row or grid[r][col] != 0:
            continue
        domain = assignments[(r, col)]
        if len(domain) == 1 and val in domain:
            return False
        
    start_row = (int)(row - row % 3)
    start_col = (int)(col - col % 3)
    #print(start_row, start_col)
    for i in range(3):
        for j in range(3):
            if row == start_row + i and col == start_col + j:
                continue
            if grid[start_row + i][start_col + j] != 0:
                continue
            domain = assignments[(start_row + i, start_col + j)]
            if len(domain) == 1 and val in domain:
                #print("failing in pt", start_row + i, start_col + j)
                return False
    return True
    
    
def backtrack_forwardcheck(grid):

    """
    Backtracking with forward check.
    """    

    if 0 not in grid:
        return grid
    
    # select a random unassigned variable: this is an assignment key.
    unassigned_variables = get_unassigned_variables(grid)
    if len(unassigned_variables) == 0:
        return None
    #print(variables_filled)
    var = random.choice(unassigned_variables)

    r = var[0]
    c = var[1]
    
    domain = [x for x in range(1,10)]
    
    #s = time.time()
    assignments = create_assignments(grid)
    #e = time.time()
    #print(e - s)
    
    while len(domain) != 0:
        value = random.choice(domain)
        #print("var", var, "value", value)
        domain.remove(value)
        if check_val(value, r, c, grid):
            
            fc_status = forwardcheck(grid, assignments, r, c, value)
            if fc_status:
                grid[r][c] = value
                
                result = backtrack_forwardcheck(grid)
                if result is not None and 0 not in result:
                    return result
                grid[r][c] = 0
            #else:
                #print("false")

# OPTION C


def minimum_remaining_values(grid, assignments) -> tuple:
    
    """
    Returns a list of variables with the fewest possible values.
    """
    unassigned_variables = get_unassigned_variables(grid)
    
    if len(unassigned_variables) == 0:
        return None

    domain_frequencies = []
    min_count = 10 # cannot exceed this
    for i in unassigned_variables:
        #print(i, assignments[i])
        cur_count = len(assignments[i])
        domain_frequencies.append((i, cur_count))
        if cur_count < min_count:
            min_count = cur_count
        
    #print(domain_frequencies)
    retvals = [x[0] for x in domain_frequencies if x[1] == min_count]
    #print(retvals)
    
    # Select a random value in retvals
    #retval = random.choice(retvals)
    #print(retval)
    return retvals
    
    
def least_constraining_value(grid, assignments: dict, 
                             row: int, col: int, domain):
    
    """
    Given a possible variable from domain, 
        selects the least constraining value.
    """
    
    #domain = assignments[(row, col)]
    retvals = []
    min_variables_constrained = 30 # This can't possibly occur.
    
    for val in domain:
        cur_count = 0
        
        # Check row, check column, check grid.
        for c in range(N):
            if c == col or grid[row][c] != 0:
                continue
            domain = assignments[(row, c)]
            if len(domain) == 1 and val in domain:
                cur_count += 1
            
        for r in range(N):
            if r == row or grid[r][col] != 0:
                continue
            domain = assignments[(r, col)]
            if len(domain) == 1 and val in domain:
                cur_count += 1
            
        start_row = (int)(row - row % 3)
        start_col = (int)(col - col % 3)
        #print(start_row, start_col)
        for i in range(3):
            for j in range(3):
                if row == start_row + i and col == start_col + j:
                    continue
                if grid[start_row + i][start_col + j] != 0:
                    continue
                domain = assignments[(start_row + i, start_col + j)]
                if len(domain) == 1 and val in domain:
                    #print("failing in pt", start_row + i, start_col + j)
                    cur_count += 1
        retvals.append((val, cur_count))
        
        if cur_count < min_variables_constrained:
            min_variables_constrained = cur_count
    
    #print("retval from lcv:", retvals)
    retval = random.choice(retvals)
    #print(retval[0])
    return retval[0]
    

def degree_heuristic(grid, assignments: dict,
                     list_of_vars: list):
    
    """
    Assign a value to the variable that is involved in the largest
    number of constraints on other unassigned variables.
    """
    
    degree_list = []
    max_degree = -1 # not possible
    
    for var in list_of_vars:
        
        #print(var)
        
        row = var[0]
        col = var[1]
        
        deg_count = 0
        
        for c in range(N):
            if c == col:
                continue
            if grid[row][c] == 0:
                deg_count += 1
            
        for r in range(N):
            if r == row:
                continue
            if grid[r][col]:
                deg_count += 1
            
        start_row = (int)(row - row % 3)
        start_col = (int)(col - col % 3)
        #print(start_row, start_col)
        for i in range(3):
            for j in range(3):
                if row == start_row + i and col == start_col + j:
                    continue
                if grid[start_row + i][start_col + j] == 0:
                    deg_count += 1
                    
        if deg_count > max_degree:
            max_degree = deg_count
                    
        degree_list.append((var, deg_count))
        
    retvals = [x[0] for x in degree_list if x[1] == max_degree]
    
    if len(retvals) > 1:
        return random.choice(retvals)
    
    return retvals[0]
        

def backtrack_fcheck_heuristics(grid):

    
    if 0 not in grid:
        return grid
    
    # Improvements: Use heuristics to change the order of assignment
    # of variables
    
    assignments = create_assignments(grid)
    
    # use MRV to obtain optimal variable
    variables = minimum_remaining_values(grid, assignments) 
    
    # select a random unassigned variable: this is an assignment key.
    
    if len(variables) > 1:
        var = degree_heuristic(grid, assignments, variables)
    else:
        var = variables[0]

    r = var[0]
    c = var[1]
    
    domain = [x for x in range(1,10)]
    
    #s = time.time()
    assignments = create_assignments(grid)
    #e = time.time()
    #print(e - s)
    
    while len(domain) != 0:
        value = least_constraining_value(grid, assignments, 
                                         r, c, domain)
        #print("var", var, "value", value)
        domain.remove(value)
        if check_val(value, r, c, grid):
            
            fc_status = forwardcheck(grid, assignments, r, c, value)
            if fc_status:
                grid[r][c] = value
                
                result = backtrack_forwardcheck(grid)
                if result is not None and 0 not in result:
                    return result
                grid[r][c] = 0
            #else:
            #    print("false")
    

def main() -> None:
    
    # Change this for different versions
    testing_A = False
    testing_B = False
    testing_C = True

    easy_str = """010900053
    040300681
    070050900
    590070040
    700805009
    020030067
    009010070
    157003090
    480002030"""
    
    med_str = """000160925
    007000180
    000080006
    780200000
    025010630
    000008017
    100070000
    062000700
    874023000"""
    
    hard_str = """
    103005400
    800000000
    000080600
    006049000
    079608240
    000210900
    005070000
    000000006
    007500103"""
    
    evil_str = """001970000
    090003008
    000402000
    610000405
    003000100
    407000029
    000705000
    200300010
    000089500"""
    
    easy = Sudoku(easy_str)
    med = Sudoku(med_str)
    hard = Sudoku(hard_str)
    evil = Sudoku(evil_str)
    
    # 12 seconds?
    if testing_A:
        
        easy_start_a = time.time()
        print(easy.grid)
        easy_sol_a = easy.solve("A") 
        print(easy_sol_a)
        easy_end_a = time.time()
        print(easy_end_a - easy_start_a)
    
        #med_start = time.time()
        #print(med.grid)
        #med_sol = med.solve("A")
        #print(med_sol)
        #med_end = time.time()
        #print(med_end - med_start)
        
    if testing_B:
        
        easy_start_b = time.time()
        easy_sol_b = easy.solve("B")
        print(easy_sol_b)
        easy_end_b = time.time()
        print(easy_end_b - easy_start_b)
        
        #med_start = time.time()
        #med_sol = med.solve("B")
        #print(med_sol)
        #med_end = time.time()
        #print(med_end - med_start)
        
    if testing_C:
        # print("here")
        # print(easy.grid)
        # easy_start_c = time.time()
        # easy_sol_c = easy.solve("C")
        # print(easy_sol_c)
        # easy_end_c = time.time()
        # print(easy_end_c - easy_start_c)
        
        # print("here")
        # print(med.grid)
        # med_start = time.time()
        # med_sol = med.solve("C")
        # print(med_sol)
        # med_end = time.time()
        # print(med_end - med_start)
        
        # print("here")
        # print(hard.grid)
        # hard_start = time.time()
        # hard_sol = hard.solve("C")
        # print(hard_sol)
        # hard_end = time.time()
        # print(hard_end - hard_start)
        
        print("here")
        print(evil.grid)
        evil_start = time.time()
        evil_sol = evil.solve("C")
        print(evil_sol)
        evil_end = time.time()
        print(evil_end - evil_start)
    
    
    # hard = Sudoku(hard_str)
    # print(hard.grid)
    # hard_sol = hard.solve("A")
    # print(hard_sol)
    
    
    #print(easy.grid.all() == easy_sol.all())
    
if __name__=="__main__":
    main()


            