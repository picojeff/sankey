# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 16:40:14 2025

@author: JSNIDER
"""

import numpy as np
import InferIntersections
from SankeyFromPairwiseIntersections import sankeyAllConnect

if __name__ == '__main__':
    n_cols = 6
    n_x = int(2**n_cols - 1)
    
    Y = InferIntersections.getY(n_cols)
    hidden_x = np.random.lognormal(4,1,size=n_x)
    b_actual = np.matmul(Y, hidden_x.transpose())
    pairwise_data = InferIntersections.PairwiseVectorToArray(b_actual, n_cols)
    
    x_calc = InferIntersections.infer_intersections(pairwise_data)
    
    print(f"hidden x: {hidden_x}")
    print(f"calculated x: {x_calc}")
    x_delta = hidden_x - x_calc
    print(f"difference: {hidden_x-x_calc}")
    
    b_calc = np.matmul(Y, x_calc)
    
    b_delta = b_actual - b_calc
    
    print(f"B delta: {b_delta}")

    sankeyAllConnect(n_cols, hidden_x, colwidth=0.05, labels=None, showterminal=False)
    
    