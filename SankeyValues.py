# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 19:38:53 2025

@author: jeff
"""

import numpy as np
#from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go

def corder(b, c, n): # reorder bits
    bb = [(b>>i)&1 for i in range(n)]
    v = bb.pop(c)
    bb = [v] + bb
    return(bb)
    
def colsort(blist, c, ncols): # sort numbers by column c in their binary representation
    return(sorted(blist, key=lambda b: corder(b,c,ncols), reverse=True))

n_cols = 3
n_x = int(2**n_cols - 1)
half_x = 1<<(n_cols-1)
n_given = int(n_cols*(n_cols+1)/2)

Y_b = np.zeros((n_given,n_x), dtype=int)

for i in range(n_cols):
    """ A row = e.g., 1111000 for n=3, 111111110000000 for n=4
    B row = e.g., 1100110 for n=3, 111100001111000 for n=4 """
    v = n_x - i
    for j in range(n_x):
        b = (j >> (n_cols-1-i)) & 1
        Y_b[i, j] = 1 - b

rown = n_cols
for i in range(n_cols):
    for j in range(i+1,n_cols):
        """ AB row = intersection of A row and B row """
        Y_b[rown] = Y_b[i] & Y_b[j]
        rown += 1

n_fixed_rows = n_given
n_free_rows = n_x - n_given

Y = Y_b.astype(float)

YYT = np.matmul(Y,Y.transpose())
YYTI = np.linalg.inv(YYT)

hidden_x = np.random.lognormal(2,1,size=n_x)
b_actual = np.matmul(Y, hidden_x.transpose())

x_calc = np.matmul(np.matmul(Y.transpose(),YYTI),b_actual)

print(f"hidden x: {hidden_x}")
print(f"calculated x: {x_calc}")
x_delta = hidden_x-x_calc
print(f"difference: {hidden_x-x_calc}")

b_calc = np.matmul(Y, x_calc)

b_delta = b_actual - b_calc

print(f"B delta: {b_delta}")

if False: # one connection between nodes
    """ build up node, source, target, and value lists """
    
    nodes = []
    nodecolor = []
    
    for i in range(n_cols):
        nodes.append(str(i))
        nodes.append(f"not-{i}")
        nodecolor.append('blue')
        nodecolor.append('red')
        
    linkvals = np.zeros((2*n_cols,2*n_cols))
    
    for i in range(n_x):
        for from_col in range(n_cols-1):
            to_col = from_col + 1
            isin0 = ((i+1) >> (n_cols-1-from_col)) & 1
            isin1 = ((i+1) >> (n_cols-1-to_col)) & 1
            from_idx = 2*from_col + 1 - isin0
            to_idx = 2*to_col + 1 - isin1
            linkvals[from_idx,to_idx] += hidden_x[i]
    
    source,target,value = [],[],[]
    for from_idx in range(2*n_cols-1):
        for to_idx in range(from_idx+1,2*n_cols):
            if linkvals[from_idx,to_idx] > 0:
                source.append(from_idx)
                target.append(to_idx)
                value.append(linkvals[from_idx,to_idx])
    
    print(list(zip(source,target,value)))
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0),
          label = [str(i) for i in range(n_cols)],
          color = nodecolor,
        ),
        link = dict(
          source = source,
          target = target,
          value = value
      ))])
    
    #fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    #fig.show()
    
    plotly.offline.plot(fig)

if True: # all connections independent across width
    """ Build up chart showing all distinct membership intersections """
    nodes = []
    nodecolor = []
    node_x = []
    node_y = []
    
    ytot = sum(x_calc)
    
    for c in range(n_cols):
        blist = colsort([_d for _d in range(1,n_x+1)], c, n_cols)
        
        ypos = [0 for i in range(n_x)]
        ysum = 0
        for i in range(n_x):
            ypos[i] = ysum/ytot
            ysum += x_calc[n_x-blist[i]]
        
        for i in range(n_x):
            bval = n_x - i
            isin0 = ((bval) >> (n_cols-1-c)) & 1
            xp = c/(n_cols-1) * 0.98 + 0.01
            yp = ypos[i]/ytot * 0.98 + 0.01
            node_x.append(xp)
            node_y.append(yp)
            nodes.append(f"{c} {bval:03b} {(node_x[-1],node_y[-1])})")
            nodecolor.append('blue' if isin0 else 'red')
    
    print(f"node xy {[i for i in zip(node_x,node_y)]}")
    source,target,value = [],[],[]
    
    for i in range(n_x):
        bval = n_x - i
        for from_col in range(n_cols-1):
            to_col = from_col + 1
            isin0 = ((bval) >> (n_cols-1-from_col)) & 1
            isin1 = ((bval) >> (n_cols-1-to_col)) & 1
            from_idx = n_x*from_col + i
            to_idx = n_x*to_col + i
            source.append(from_idx)
            target.append(to_idx)
            value.append(x_calc[i])
    
    print(list(zip(source,target,value)))
    
    fig = go.Figure(data=[go.Sankey(
        arrangement = 'snap',
        node = dict(
          pad = 0,
          thickness = 20,
          line = dict(color = "black", width = 0),
          label = nodes,
          color = nodecolor,
          x = node_x,
          y = node_y
        ),
        link = dict(
          source = source,
          target = target,
          value = value
      ))])
    
    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    #fig.show()
    
    plotly.offline.plot(fig)


# U,S,VT = np.linalg.svd(Y)

# """ tack on extra orthogonal rows to make a square nonsingular matrix """
# # XXX there should be a theoretical approach that makes this rank-testing unnecessary XXX
# nexttry = -1
# cur_rank = np.linalg.matrix_rank(Y)
# new_rank = cur_rank
# for i in range(n_free_rows):
#     while new_rank == cur_rank:
#         Y[-1-i] = U[nexttry]
#         nexttry -= 1
#         new_rank = np.linalg.matrix_rank(Y)
#     cur_rank = new_rank
    

# print(f"Y: {Y}")
# print(f"Y has size {len(Y)} and rank {np.linalg.matrix_rank(Y)}")

# if n_cols == 3:
#     # hidden values
#     A111 = 5
#     A110 = 10
#     A101 = 15
#     A100 = 5
#     A011 = 20
#     A010 = 6
#     A001 = 10
    
#     """ make a plot over the range of the one degree of freedom """
#     x = np.linspace(-40,100)
#     y = np.zeros(len(x))
    
#     for i,xv in enumerate(x):
#         b_vals = np.array([A111+A110+A101+A100,
#                            A111+A110+A011+A010,
#                            A111+A101+A011+A001,
#                            A111+A110,
#                            A111+A101,
#                            A111+A011,
#                            xv]).transpose()
        
#         A_calc = np.linalg.solve(Y, b_vals)
#         y[i] = sum(A_calc**2)
        
#     plt.plot(x,y)

# """ construct hidden variables which we try to reconstruct from
# the few pairwise intersections we are able to see """

# n_actual = 2**n_cols # actual number of hidden variables

# # array of the hidden variables
# hidden_x = np.random.lognormal(1,1,size=n_x) # every intersection across all cols
# print(f"hidden values: {hidden_x}")

# """ the first n_fixed_rows will be what we receive as pairwise information """
# b_actual = np.matmul(Y, hidden_x.transpose())

# print(f"b values: {b_actual}")

# print(f"{n_free_rows} degrees of freedom")

# b_vals = [i for i in b_actual]

# """ first step is to plug in random values for the free rows and see what we get """
# val_range = max(abs(b_actual[0:n_given]))
# for i in range(n_free_rows):
#     b_vals[-1-i] = np.random.random()*val_range

# print(f"testing b values: {b_vals}")

# x_calc = np.linalg.solve(Y, b_vals)
# print(f"calculated values: {x_calc}")
   
