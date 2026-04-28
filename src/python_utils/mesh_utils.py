# Libraries {{{
import numpy as np
from mpi4py import MPI
import gmsh
from pdb import set_trace

from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d

from shapely.geometry import Point
# }}}

# Functions {{{
# Make sand-clock {{{
def MakeSandClock(radius : float, lc : float, meshOrder = 1,
                  centre = (0.0, 0.0)):
    # Initialisation
    gmsh.initialize()
    model = gmsh.model
    geo = model.occ
    model.add("sand-clock")
    # Create points
    rt  = geo.addPoint(centre[0] + radius , centre[1] + 2.0*radius , 0.0 , lc)
    ct  = geo.addPoint(centre[0]          , centre[1] + 2.0*radius , 0.0 , lc)
    lt  = geo.addPoint(centre[0] - radius , centre[1] + 2.0*radius , 0.0 , lc)
    lmt = geo.addPoint(centre[0] - radius , centre[1] + radius     , 0.0 , lc)
    cc  = geo.addPoint(centre[0]          , centre[1]              , 0.0 , lc)
    lmb = geo.addPoint(centre[0] - radius , centre[1] - radius     , 0.0 , lc)
    lb  = geo.addPoint(centre[0] - radius , centre[1] - 2.0*radius , 0.0 , lc)
    cb  = geo.addPoint(centre[0]          , centre[1] - 2.0*radius , 0.0 , lc)
    rb  = geo.addPoint(centre[0] + radius , centre[1] - 2.0*radius , 0.0 , lc)
    rmb = geo.addPoint(centre[0] + radius , centre[1] - radius , 0.0 , lc)
    rmt = geo.addPoint(centre[0] + radius , centre[1] + radius , 0.0 , lc)
    # Create spline
    # spline = geo.addBSpline([rt, ct, lt, lmt, cc, lmb, lb, cb, rb, rmb, cc, rmt, rt])
    spline = geo.addBSpline([rt, rmt, cc, rmb, rb, cb, lb, lmb, cc, lmt, lt, ct, rt])
    # Create loop
    loop = geo.addCurveLoop([spline])
    # Synchronise
    geo.synchronize()
    # Groups
    gr_gamma = model.addPhysicalGroup(1, [spline])
    # Mesh
    model.mesh.generate(dim = 1)
    model.mesh.setOrder(meshOrder)
    # gmsh.fltk.run()
    return model
# }}}
# Make circle mesh {{{
def MakeCircle(radius : float, lc : float, meshOrder = 1,
               centre = (0.0, 0.0)):
    # Initialisation
    gmsh.initialize()
    model = gmsh.model
    geo = model.occ
    model.add("circle")
    # Create points
    cc = geo.addPoint(centre[0], centre[1], 0.0, lc)
    rp = geo.addPoint(centre[0] + radius, centre[1], 0.0, lc)
    lp = geo.addPoint(centre[0] - radius, centre[1], 0.0, lc)
    # Create lines
    topLine = geo.addCircleArc(rp, cc, lp)
    botLine = geo.addCircleArc(lp, cc, rp)
    # Create loop
    loop = geo.addCurveLoop([topLine, botLine])
    # Synchronise
    geo.synchronize()
    # Groups
    gr_gamma = model.addPhysicalGroup(1, [topLine, botLine])
    # Mesh
    model.mesh.generate(dim = 1)
    model.mesh.setOrder(meshOrder)
    # gmsh.fltk.run()
    return model
# }}}
# Create ordered node list {{{
def OrderNodeList(startNode, endNode, nodes, numEles):
    # Get the total number of elements
    # Initialisation of node list
    nodeList = [startNode]
    oldNode = startNode
    # Find the element containing the old node
    newEle = np.argwhere(nodes == oldNode)[0, 0]
    # Get the nodes of the element (start, end, middle)
    ns, ne, nm = nodes[newEle, :]
    # If the middle node is valid add it to the node list
    if nm != -1:
        nodeList.append(nm)
    # Find which node of the new element is the new node
    if oldNode == ns:
        newNode = ne
    elif oldNode == ne:
        newNode = ns
    else:
        raise ValueError("Invalid nodes")
    # Add the new node to the node list
    nodeList.append(newNode)
    # Repeat the process adding nodes following their connection on edges
    for k1 in range(numEles - 1):
        # Update old values
        oldNode = newNode
        oldEle = newEle
        # Find new element
        ele1, ele2 = np.argwhere(nodes == oldNode)[:, 0]
        if oldEle == ele1:
            newEle = ele2
        elif oldEle == ele2:
            newEle = ele1
        else:
            raise ValueError("Invalid elements")
        # Get the nodes of the element (start, end, middle)
        ns, ne, nm = nodes[newEle, :]
        # If the middle node is valid add it to the node list
        if nm != -1:
            nodeList.append(nm)
        # Find which node of the new element is the new node
        if oldNode == ns:
            newNode = ne
        elif oldNode == ne:
            newNode = ns
        else:
            raise ValueError("Invalid nodes")
        nodeList.append(newNode)
    nodeList = np.array(nodeList)
    return nodeList
# }}}
# Equidistribute mesh {{{
def EquidistributeMesh(coords, bc_type = "periodic", inSamples = 100, optimal = True):
    # Cumulative length
    spl_x, spl_y, cumSpline_length = ArcLengthSpline(coords, bc_type = bc_type,
                                                     inSamples = inSamples)
    spline_length = cumSpline_length[-1]
    numPoints = len(cumSpline_length) - 1 # No repeated points
    para = np.arange(numPoints + 1) # Parameter
    # Optimal lengths
    dumLengths = np.linspace(0.0, spline_length, numPoints + 1)
    if optimal:
        disLengths = (dumLengths - cumSpline_length)[:-1]
        delLength = -np.sum(disLengths)/numPoints
        newLengths = dumLengths - delLength
    else:
        newLengths = dumLengths
    # New positions
    interp_func = interp1d(cumSpline_length, para, kind = 'linear',
                           fill_value = "extrapolate")
    newPara = interp_func(newLengths)
    new_x = spl_x(newPara)
    new_y = spl_y(newPara)
    # New coords
    equiCoords = np.column_stack([new_x, new_y])
    return equiCoords
# }}}
# Compute arc length spline {{{
def ArcLengthSpline(coords, bc_type = "periodic", inSamples = 1000):
    # Get data to define parameterised spline
    x = coords[:, 0]
    y = coords[:, 1]
    numPoints = len(coords) - 1 # No repeated points
    para = np.arange(numPoints + 1) # Parameter
    # Define spline
    spl_x = make_interp_spline(para, x, bc_type = bc_type)
    spl_y = make_interp_spline(para, y, bc_type = bc_type)
    # Segment length
    para_segments = np.linspace(0, numPoints, inSamples*numPoints)
    dpara = (numPoints)/(inSamples*numPoints)
    dx = spl_x(para_segments, 1)
    dy = spl_y(para_segments, 1)
    step_length_point = np.sqrt(dx**2.0 + dy**2.0)*dpara
    step_length_point_half_left = step_length_point/2.0
    step_length_point_half_right = np.concatenate([step_length_point_half_left[1:],
                                                   np.array([step_length_point_half_left[0]])])
    step_length = step_length_point_half_left + step_length_point_half_right
    cumSpline_length_fine = np.cumsum(step_length)
    cumSpline_length = np.concatenate([np.zeros(1),
                                       cumSpline_length_fine[(para*inSamples - 1)[1:]]])
    return spl_x, spl_y, cumSpline_length

# }}}
# Compute Chebyshev center {{{
def ChebyshevCenter(polygon, tol = 1.0e-3, max_ites = 100):
    # Signed distance
    def signedDist(c):
        pt = Point(c)
        inside = polygon.contains(pt)
        dist = polygon.exterior.distance(pt)
        dist = -dist if not inside else dist
        return dist
    # Get polygon bounds
    coords = np.array(polygon.exterior.coords)
    xmin = coords[:, 0].min()
    xmax = coords[:, 0].max()
    ymin = coords[:, 1].min()
    ymax = coords[:, 1].max()
    xlen = (xmax - xmin)/2.0
    ylen = (ymax - ymin)/2.0
    square_len = np.sqrt(xlen**2.0 + ylen**2.0)
    centroid = np.array((xmin + xmax, ymin + ymax))/2.0
    squares = [(centroid, square_len, xlen, ylen)]
    potential_distances = np.array([signedDist(centroid) + square_len/2.0])
    for k1 in range(max_ites):
        # Find most likely square to contain the center
        max_square_id = np.argmax(potential_distances)
        centroid_k1, square_len_k1, xlen_k1, ylen_k1 = squares[max_square_id]
        if square_len_k1 < tol:
            break
        # Set the new centroids
        xcen, ycen = centroid_k1
        xlen = xlen_k1/2.0
        ylen = ylen_k1/2.0
        square_len = np.sqrt(xlen**2.0 + ylen**2.0)
        tl = np.array([xcen - 0.5*xlen, ycen - 0.5*ylen])
        tr = np.array([xcen + 0.5*xlen, ycen - 0.5*ylen])
        bl = np.array([xcen - 0.5*xlen, ycen + 0.5*ylen])
        br = np.array([xcen + 0.5*xlen, ycen + 0.5*ylen])
        # Make new squares
        new_squares = [(tl, square_len, xlen, ylen),
                       (tr, square_len, xlen, ylen),
                       (bl, square_len, xlen, ylen),
                       (br, square_len, xlen, ylen)]
        # Compute new potential distances
        new_potential_distances = np.array([signedDist(tl) + square_len/2.0,
                                            signedDist(tr) + square_len/2.0,
                                            signedDist(bl) + square_len/2.0,
                                            signedDist(br) + square_len/2.0])
        # Remove head square and add new squares
        squares.pop(max_square_id)
        squares += new_squares
        potential_distances = np.concatenate([np.delete(potential_distances, max_square_id),
                                              new_potential_distances])
    # Select best centre
    max_square_id = np.argmax(potential_distances)
    centroid = squares[max_square_id][0]
    return centroid_k1
# }}}
# }}}
