# This file is for setting up the boundary conditions ("the table with
# stuff like ropes, boxes, etc.") and visualizing them

import numpy
from shapely import geometry
from shapely import affinity
from matplotlib import pyplot
from descartes import PolygonPatch
import cPickle

# the boundary of the whole scene

frame = geometry.box(0.0, 0.0, 10.0, 10.0)

# objects on the table

left1 = geometry.box(1, 1, 4, 6)
right1 = geometry.box(6, 1, 8, 6)

# the transformed objects (transformations are given by specifying a
# new coordinate for every edge in the polygon)

left2 = left1
right2 = affinity.translate(right1, xoff = 0.0, yoff = 2)

# drawing the table

def add_object(polygon, color):
    patch = PolygonPatch(polygon, facecolor=color, alpha=0.5, zorder=2)
    ax.add_patch(patch)

fig = pyplot.figure(1, dpi=90)
ax = fig.add_subplot(111)

xlist = map(lambda x: x[0], list(frame.exterior.coords))
ylist = map(lambda x: x[1], list(frame.exterior.coords))
ax.set_xlim(min(xlist), max(xlist))
ax.set_ylim(min(ylist), max(ylist))
ax.set_aspect(1)

add_object(left1, '#999999')
add_object(right1, '#999999')
add_object(left2, '#6699cc')
add_object(right2, '#6699cc')

pyplot.title("The table")

pyplot.show()

# serialize stuff into file for reading out

obj = {'bounds' : 0, 'left' : [], 'right' : []}

# first have to orient the polygons properly

left1 = geometry.polygon.orient(left1)
right1 = geometry.polygon.orient(right1)
left2 = geometry.polygon.orient(left2)
right2 = geometry.polygon.orient(right2)

obj['bounds'] = frame.bounds
obj['left'] = []
obj['left'].append(list(left1.exterior.coords))
obj['left'].append(list(left2.exterior.coords))
obj['right'] = []
obj['right'].append(list(right1.exterior.coords))
obj['right'].append(list(right2.exterior.coords))

with open("normals.data", 'w') as f:
    cPickle.dump(obj, f)

