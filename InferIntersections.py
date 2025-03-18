# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 19:38:53 2025

@author: jeff
"""

import numpy as np

def getY(n_cols):
    n_pairs = int(n_cols*(n_cols+1)/2)
    n_x = int(2**n_cols - 1)

    Y_b = np.zeros((n_pairs,n_x), dtype=int)
    
    for i in range(n_cols):
        """ A row = e.g., 1111000 for n=3, 111111110000000 for n=4
        B row = e.g., 1100110 for n=3, 111100001111000 for n=4 """
        for j in range(n_x):
            b = (j >> (n_cols-1-i)) & 1
            Y_b[i, j] = 1 - b
    
    rown = n_cols
    for i in range(n_cols):
        for j in range(i+1,n_cols):
            """ AB row = intersection of A row and B row """
            Y_b[rown] = Y_b[i] & Y_b[j]
            rown += 1
    
    Y = Y_b.astype(float)
    
    return(Y)


def PairwiseArrayToVector(pairwise_data):
    n_cols = len(pairwise_data)
    n_pairs = int(n_cols*(n_cols+1)/2)

    b_given = np.zeros(n_pairs)
    for i in range(n_cols):
        b_given[i] = pairwise_data[i,i]
    
    nextrow = n_cols
    for i in range(n_cols):
        for j in range(i+1,n_cols):
            b_given[nextrow] = pairwise_data[i,j]
            nextrow += 1

    return(b_given)    

def PairwiseVectorToArray(b_actual, n_cols):
    b_matrix = np.zeros((n_cols,n_cols))
    for i in range(n_cols):
        b_matrix[i,i] = b_actual[i]
        
    nextrow = n_cols
    for i in range(n_cols):
        for j in range(i+1,n_cols):
            b_matrix[i,j] = b_actual[nextrow]
            nextrow += 1
    
    return(b_matrix)


""" infer inner intersections of sets given only pairwise intersections.
Input is list of lists or square numpy array showing pairwise information
in the upper right triangle. """
def infer_intersections(pairwise_data):
    n_cols = len(pairwise_data)
    
    b_given = PairwiseArrayToVector(pairwise_data)

    Y = getY(n_cols)
    
    YYT = np.matmul(Y,Y.transpose())
    YYTI = np.linalg.inv(YYT)
    YTYYTI = np.matmul(Y.transpose(),YYTI)
    x_calc = np.matmul(YTYYTI, b_given)

    return(x_calc)

if __name__ == '__main__':
    
    n_cols = 3
    n_x = int(2**n_cols - 1)
    
    hidden_x = np.random.lognormal(2,1,size=n_x)
    
    Y = getY(n_cols)
    b_actual = np.matmul(Y, hidden_x.transpose())
    
    b_matrix = PairwiseVectorToArray(b_actual, n_cols)
    
    x_calc = infer_intersections(b_matrix)
    
    print(f"hidden x: {hidden_x}")
    print(f"calculated x: {x_calc}")
    x_delta = hidden_x-x_calc
    print(f"difference: {hidden_x-x_calc}")
    
    b_calc = np.matmul(Y, x_calc)
    
    b_delta = b_actual - b_calc
    
    print(f"B delta: {b_delta}")
    
