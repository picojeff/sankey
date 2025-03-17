# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:54:46 2025

@author: jeff
"""

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def makelink(ax, p0, p1, height, facecolor, alpha=1.0):
    x0,y0 = p0
    x1,y1 = p1
    if type(height) == tuple:
        h0,h1 = height
    else:
        h0,h1 = height,height
    midx = (x0+x1)/2
    verts = [
       (x0, y0),
       (midx, y0),  
       (midx, y1),  
       (x1, y1),  
       (x1, y1-h1),   
       (midx, y1-h1),  
       (midx, y0-h0),  
       (x0, y0-h0),
       (x0, y0)
    ]
    
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY
    ]
    
    path = Path(verts, codes)

    patch = patches.PathPatch(path, facecolor=facecolor, lw=1, ec='white', alpha=alpha)
    ax.add_patch(patch)

fig, ax = plt.subplots()


makelink(ax, (0, 1), (1, 0.4), 0.5, 'grey', alpha=0.5)
makelink(ax, (0, 0.4), (1, 1), 0.5, 'grey', alpha=0.5)

#xs, ys = zip(*verts)
#ax.plot(xs, ys, 'x--', lw=2, color='black', ms=10)

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

plt.show()