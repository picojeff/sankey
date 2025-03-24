# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 19:38:53 2025

@author: jeff
"""

import numpy as np
from matplotlib import pyplot as plt

import Sankey

def corder(b, c, n): # reorder bits
    bb = [(b>>i)&1 for i in range(n-1,-1,-1)]
    for j in range(1,c+1):
        v = bb.pop(j)
        bb = [v] + bb
    return(bb)
    
def colsort(blist, c, ncols, reverse=True): # sort numbers by column c in their binary representation
    return(sorted(blist, key=lambda b: corder(b,c,ncols), reverse=reverse))

def sankeySingleConnect(n_cols, x_vals): # one connection between nodes
    """ build up node, source, target, and value lists """
    
    n_x = int(2**n_cols - 1)

    nodes = []
    nodecolor = []
    node_x,node_y = [],[]
    noderank = []

    linkvals = np.zeros((2*n_cols,2*n_cols))
    
    """ build list from top down """
    for i in range(n_x-1,0,-1):
        for from_col in range(n_cols-1):
            to_col = from_col + 1
            isin0 = ((i+1) >> (n_cols-1-from_col)) & 1
            isin1 = ((i+1) >> (n_cols-1-to_col)) & 1
            from_idx = 2*from_col + 1 - isin0
            to_idx = 2*to_col + 1 - isin1
            linkvals[from_idx,to_idx] += x_vals[i]

    print(f"linkvals {linkvals}")
    
    """ build nodes from top down in each column """
    for i in range(n_cols):
        nodes.append(str(i))
        nodes.append(f"not-{i}")
        nodecolor.append('blue')
        nodecolor.append('red')
        node_x.append(i/(n_cols-1))
        node_x.append(i/(n_cols-1))
        if i >= 1:
            botnodein = linkvals[2*i-2,2*i+1] + linkvals[2*i-1,2*i+1]
        else:
            botnodein = 0
        if i < n_cols-1:
            botnodeout = linkvals[2*i+1,2*i+2] + linkvals[2*i+1,2*i+3]
        else:
            botnodeout = 0
        
        node_y.append(max(botnodein,botnodeout))
        node_y.append(0)
        noderank.append(i)
        noderank.append(i)
        
    
    source,target,value = [],[],[]
    for from_idx in range(2*n_cols-1):
        for to_idx in range(2*n_cols-1,from_idx,-1):
            if linkvals[from_idx,to_idx] > 0:
                source.append(from_idx)
                target.append(to_idx)
                value.append(linkvals[from_idx,to_idx])
    
    # source.reverse()
    # target.reverse()
    # value.reverse()
    
    # print(list(zip(source,target,value)))
    
    sankey_info = dict(
        node = dict(
          width = 0.1,
          line = dict(color = "black", width = 0),
          label = [str(i) for i in range(n_cols)],
          color = nodecolor,
          x = node_x,
          y = node_y,
          rank = noderank,
        ),
        link = dict(
          source = source,
          target = target,
          value = value,
          color = 'grey',
      ))
    
    print(f"sankey info: {sankey_info}")
    #fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    #fig.show()
    
    fig, ax = plt.subplots(dpi=300, figsize=(8,5))
    Sankey.doSankey(ax, sankey_info)
    plt.show()

def sankeyAllConnect(n_cols, x_values, colwidth=None, labels=None, showterminal=True): # all connections independent across width
    """ Build up chart showing all distinct membership intersections """
    
    n_x = int(2**n_cols - 1)

    nodes = []
    nodecolor = []
    nodealpha = []
    node_x = []
    node_y = []
    noderank = []
    
    x_calc = np.array([0 if i < 0 else i for i in x_values])
    
    # XXX Technical debt here is large
    
    """ for each column, order nodes from top to bottom """
    blist = []
    blookup = np.zeros((n_cols,n_x+1), dtype=int)
    for c in range(n_cols):
        forward_list = colsort([_d for _d in range(1,n_x+1)], c, n_cols, reverse=True)
        blist.append(forward_list)
        for i,idv in enumerate(forward_list):
            blookup[c][idv] = n_x * c + i
    print(f"colsort: {blist}")
    # print(f"blookup: {blookup}")
    
    terminallookup = np.zeros((n_cols,n_x+1))
    
    """ for each column, calculate y position of node """
    ytot = sum(x_calc)
    for c in range(n_cols):
       ypos = [0 for i in range(n_x)]
       ysum = ytot
       for i in range(n_x):
           ysum -= x_calc[blist[c][i]-1]
           ypos[blist[c][i]-1] = ysum/ytot
       
       for i in range(n_x):
           bval = blist[c][i]
           isterminal = not (bval & (2**(n_cols-c)-1)) # only zeros from this column on to the right
           terminallookup[c,bval] = isterminal
           
           xp = c/(n_cols-1) * 0.98 + 0.01
           yp = (ypos[bval-1])* 0.98 + 0.01
           node_x.append(xp)
           node_y.append(yp)
           nodes.append(f"{c} {bval:03b}") # {(node_x[-1],node_y[-1])}")
           isin0 = (bval >> (n_cols-1-c)) & 1
         
           nc = 'blue' if isin0 else 'inv' if isterminal else 'red' if c == 0 else 'grey'
           if nc == 'inv':
               na = 0
               nc = 'grey'
           elif nc =='grey':
               na = 0.5
           else:
               na = 1
           nodecolor.append(nc)
           nodealpha.append(na)
           noderank.append(c)

    """ for each column, link from this node to next column """
    
    # print(f"node xy {[i for i in zip(node_x,node_y)]}")
    source,target,value,linkcolor,linkalpha = [],[],[],[],[]
    
    for i in range(1,n_x+1):
        for from_col in range(n_cols-1):
            to_col = from_col + 1
            from_idx = blookup[from_col][i]
            to_idx = blookup[to_col][i]
            source.append(from_idx)
            target.append(to_idx)
            value.append(x_calc[i-1])
            linkcolor.append('grey')
            isterminal = terminallookup[to_col,i]
            linkalpha.append(0.5 if not isterminal else 0)
            
    
    # print(list(zip(source,target,value)))
    # source.reverse()
    # target.reverse()
    # value.reverse()
    
    if labels and labels == 'nodes':
        labels = nodes
    if not colwidth:
        colwidth = 0.1
        
    sankey_info = dict(
        arrangement = 'snap',
        node = dict(
            width = colwidth,
            line = dict(color = "black", width = 0),
            label = labels,
            color = nodecolor,
            alpha = nodealpha,
            x = node_x,
            y = node_y,
            rank = noderank,
            yscale = False,
          ),
          link = dict(
            source = source,
            target = target,
            value = value,
            scale = True,
            color = linkcolor,
            alpha = linkalpha,
      ))
    
    print(f"sankey info {sankey_info}")
    fig, ax = plt.subplots(dpi=600, figsize=(8,5))
    Sankey.doSankey(ax, sankey_info)
    plt.show()

