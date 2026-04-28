# Libraries {{{
import numpy as np
from pdb import set_trace
from shapely import Polygon
# }}}

# ECM Points {{{
def ECMPoints(nrows, ncols, length, y_start, shift_factor):
    ecm_points = np.empty((0, 2))
    x_start = -length/2.0
    delta_x = length/(ncols - 1)
    delta_y = length/(nrows - 1)
    shift = delta_x*shift_factor
    for i in range(nrows):
        for j in range(ncols):
            x = x_start + j*delta_x
            y = y_start + i*delta_y
            if (i%2 == 0):
                x += shift
            ecm_points = np.vstack((ecm_points, np.array([x, y])))
    return ecm_points
# }}}

# ECM from list {{{
def ECM_from_list(point_list):
    return np.array(point_list)
# }}}

# ECM polygons {{{
def ECMPolygons(list_poly_coords):
    polygons = []
    for coords in list_poly_coords:
        array = np.array(coords)
        xCoor = array[:, 0]
        yCoor = array[:, 1]
        polygons.append(Polygon(zip(xCoor, yCoor)))
    return polygons
# }}}
