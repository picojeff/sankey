# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 11:54:46 2025

@author: jeff
"""

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

def makelink(ax, p0, p1, height, facecolor, alpha=1.0, ec='white', lw=0):
    x0,y0 = p0
    x1,y1 = p1
    if type(height) == tuple:
        h0,h1 = height
    else:
        h0,h1 = height,height
    ctlwid = 0.3
    midx0 = x0+(x1-x0)*ctlwid
    midx1 = x1-(x1-x0)*ctlwid
    verts = [
       (x0, y0),
       (midx0, y0),  
       (midx1, y1),  
       (x1, y1),  
       (x1, y1+h1),   
       (midx1, y1+h1),  
       (midx0, y0+h0),  
       (x0, y0+h0),
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

    patch = patches.PathPatch(path, facecolor=facecolor, lw=lw, ec=ec, alpha=alpha)
    ax.add_patch(patch)

def makebox(ax, p0, s0, facecolor, alpha=1.0, ec='white', lw=0):
    x0,y0 = p0
    width,height = s0
    x1 = x0 + width
    y1 = y0 + height
    verts = [
       (x0, y0),
       (x1, y0),  
       (x1, y1),  
       (x0, y1),  
       (x0, y0)
    ]
    
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY
    ]
    
    path = Path(verts, codes)

    patch = patches.PathPatch(path, facecolor=facecolor, lw=lw, ec=ec, alpha=alpha)
    ax.add_patch(patch)

class Box():
    def __init__(self, x=0, y=0, rank=0, width=0.1, height=0.1, color='blue', alpha=1.0):
        self.input_x = x
        self.input_y = y
        self.input_width = width
        self.input_height = height
        self.rank = rank
        self.color = color
        self.alpha = alpha
        self.inlink_total = 0
        self.outlink_total = 0
        self.bot_out = 0
        self.top_in = 0

class Link():
    def __init__(self, source=0, target=1, value=1, color='grey', alpha=0.5):
        self.source = source
        self.target = target
        self.in_value = value
        self.color = color
        self.alpha = alpha
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.height = 0

def makelistifnot(v, l):
    if type(v) != list:
        v = [v for i in range(l)]
    else:
        while len(v) < l:
            v.append(v[0])
    return(v)

def makeSankey(info):
    node = info['node']
    link = info['link']
    
    n_nodes = max([len(node[i]) for i in ['label','x','y','color','alpha','width'] if i in node and type(node[i]) in {list, tuple}])
    n_links = len(link['source'])

    for td in [('alpha',1.0),('height',None)]:
        if td[0] not in node:
            node[td[0]] = td[1]
        
    for thing in ['width', 'height', 'label', 'color', 'alpha']:
        node[thing] = makelistifnot(node[thing], n_nodes)

    for td in [('alpha',0.5),('label','')]:
        if td[0] not in link:
            link[td[0]] = td[1]

    for thing in ['value', 'label', 'alpha', 'color']:
        link[thing] = makelistifnot(link[thing], n_links)
    
    
    boxes = ['' for i in range(n_nodes)]
    for i in range(n_nodes):
        boxes[i] = Box(x = node['x'][i],
                       y = node['y'][i],
                       width = node['width'][i],
                       height = node['height'][i],
                       rank = node['rank'][i],
                       color = node['color'][i],
                       alpha = node['alpha'][i]
                   )
    
    links = ['' for i in range(n_links)]
    for i in range(n_links):
        links[i] = Link(source = link['source'][i],
                        target = link['target'][i],
                        value = link['value'][i],
                        color = link['color'][i],
                        alpha = link['alpha'][i]
                    )
    
    """ set node left and right x values """
    minx, maxx = min(node['x']), max(node['x'])
    maxrank = max([box.rank for box in boxes])
    maxwidthinlastrank = max([box.input_width for box in boxes if box.rank == maxrank])
    xscale = (maxx-minx)/(1-maxwidthinlastrank)
    
    for box in boxes:
        box.x0 = (box.input_x - minx) / xscale
        box.x1 = box.x0 + box.input_width
    
    """ build total node into- and outof- """
    for link in links:
        boxes[link.source].outlink_total += link.in_value
        boxes[link.target].inlink_total += link.in_value
    
    """ set node top and bottom y values """
    ranktotals = [0 for _d in set(node['rank'])]
    for box in boxes:
        ranktotals[box.rank] += max(box.outlink_total, box.inlink_total)
    
    linkscale = max(ranktotals)
    for link in links:
        link.height = link.in_value / linkscale
    
    miny = 1e100
    for box in boxes:
        box.height = max(box.outlink_total, box.inlink_total) / linkscale
        miny = min(miny, box.input_y)
    
    yscale = max([(box.input_y-miny)/(1-box.height) for box in boxes])
    for box in boxes:
        box.y0 = (box.input_y - miny) / yscale
        box.y1 = box.y0 + box.height
        box.top_in = box.height
    
    """ build links """
    for link in links:
        frombox = boxes[link.source]
        tobox = boxes[link.target]
        
        link.y0 = frombox.y0 + frombox.bot_out
        frombox.bot_out += link.height

        tobox.top_in -= link.height
        link.y1 = tobox.y0 + tobox.top_in
        
        link.x0 = frombox.x1
        link.x1 = tobox.x0
        
    
    return(boxes, links)

def doSankey(ax, sankey_info):
    boxes, links = makeSankey(sankey_info)
    

    for box in boxes:
        makebox(ax, (box.x0,box.y0), (box.x1-box.x0,box.y1-box.y0), box.color, alpha=box.alpha)
    
    for link in links:
        makelink(ax, (link.x0, link.y0), (link.x1, link.y1), link.height, link.color, link.alpha)

    xbounds = (0,1)
    ybounds = (0,1)
    #xs, ys = zip(*verts)
    #ax.plot(xs, ys, 'x--', lw=2, color='black', ms=10)
    
    ax.set_xlim(xbounds[0]-0.1, xbounds[1]+0.1)
    ax.set_ylim(ybounds[0]-0.1, ybounds[1]+0.1)
    ax.set_xticks([])
    ax.set_yticks([])

if __name__ == '__main__':
    """ what we need to pass to object generators: """
    sankey_info = dict(node = dict(
            label = list("abcde"),
            color = ['blue', 'red', 'orange', 'grey', 'orange'],
            rank = [0, 0, 1, 1, 2],
            x = [0, 0, 1, 1, 2],
            y = [1.5, 0, 2, 0, 1],
            alpha = 0.5,
            width = 0.05 # fraction of total image width
        ),
        link = dict(
            source = [0, 0, 1, 1, 2],
            target = [3, 2, 3, 2, 4],
            value = [2, 1, 2, 1, 2],
            color = ['grey', 'blue', 'grey', 'red', 'orange']
        ),
    )
    
    fig, ax = plt.subplots()
    doSankey(ax, sankey_info)
    plt.show()    
