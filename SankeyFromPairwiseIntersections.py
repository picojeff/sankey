# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 19:38:53 2025

@author: jeff
"""

import numpy as np
from matplotlib import pyplot as plt

import InferIntersections
import Sankey

def corder(b, c, n): # reorder bits
    bb = [(b>>i)&1 for i in range(n)]
    v = bb.pop(c)
    bb = [v] + bb
    return(bb)
    
def colsort(blist, c, ncols): # sort numbers by column c in their binary representation
    return(sorted(blist, key=lambda b: corder(b,c,ncols), reverse=True))


n_cols = 3
n_x = int(2**n_cols - 1)

Y = InferIntersections.getY(n_cols)
hidden_x = np.random.lognormal(2,1,size=n_x)
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

if True: # one connection between nodes
    """ build up node, source, target, and value lists """
    
    nodes = []
    nodecolor = []
    nodex,nodey = [],[]
    noderank = []
    
    for i in range(n_cols):
        nodes.append(str(i))
        nodes.append(f"not-{i}")
        nodecolor.append('blue')
        nodecolor.append('red')
        nodex.append(i/(n_cols-1))
        nodex.append(i/(n_cols-1))
        nodey.append(1)
        nodey.append(0)
        noderank.append(i)
        noderank.append(i)
        
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
    
    sankey_info = dict(
        node = dict(
          width = 0.1,
          line = dict(color = "black", width = 0),
          label = [str(i) for i in range(n_cols)],
          color = nodecolor,
          x = nodex,
          y = nodey,
          rank = noderank,
        ),
        link = dict(
          source = source,
          target = target,
          value = value,
          color = 'grey',
      ))
    
    print(sankey_info)
    #fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    #fig.show()
    
    fig, ax = plt.subplots(dpi=300, figsize=(8,5))
    Sankey.doSankey(ax, sankey_info)
    plt.plot()

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

