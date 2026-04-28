# Libraries {{{
import dolfinx
from dolfinx import plot, fem
import pyvista
import vtk
import gmsh
import ufl
import inspect
import sys
import os
from ufl import (TestFunction, TrialFunction, Identity, grad, inner, det,
                 inv, tr, as_vector, outer, derivative, dev, sqrt)

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
    "font.size" : 9})
plt.close("all")

from pdb import set_trace
# }}}

# mprint {{{
# From: https://github.com/ericstewart36/finite_viscoelasticity/blob/main/FV09_NBR_bushing_shear_MPI.py
# this forces the program to still print (but only from one CPU)
# when run in parallel.
def mprint(*argv, rank = 0):
    if rank==0:
        out = ""
        for arg in argv:
            out = out + str(argv)
        print(out, flush = True)
# }}}
# From vector to matrix {{{
def FromVectorToMatrix(vector :  np.ndarray, numComp : int):
    # Check compatibility
    vSize = vector.shape[0]
    if not vSize%numComp == 0:
        raise("Incompatible vector size with number of components")
    # Initialise matrix
    numRows = int(vSize/numComp)
    matrix = np.zeros((numRows, numComp))
    # Assign values
    for k1 in range(numRows):
        for k2 in range(numComp):
            matrix[k1, k2] = vector[numComp*k1 + k2]
    return matrix
# }}}
# Combine classes {{{
def CombineClasses(classes, module):
    # New class name
    # Build class name automatically
    newClassName = "_".join(classes)
    # Get classes
    CLASS_REGISTRY = {
            name : cls for name, cls in inspect.getmembers(module, inspect.isclass)
            }
    # Make combined class list
    bases = tuple(CLASS_REGISTRY[name] for name in classes)
    return type(newClassName, bases, {})
# }}}
# Plot circles {{{
def PlotCircles(points, radius, filename, num_points = 20):
    gmsh.initialize()
    gmsh.model.add("circles")
    geo = gmsh.model.occ

    lc = radius/num_points
    surfaces = []

    for (cx, cy) in points:
        c = geo.addPoint(cx, cy, 0, lc)

        pts = [
            geo.addPoint(cx + radius, cy, 0, lc),
            geo.addPoint(cx, cy + radius, 0, lc),
            geo.addPoint(cx - radius, cy, 0, lc),
            geo.addPoint(cx, cy - radius, 0, lc),
        ]

        arcs = [
            geo.addCircleArc(pts[0], c, pts[1]),
            geo.addCircleArc(pts[1], c, pts[2]),
            geo.addCircleArc(pts[2], c, pts[3]),
            geo.addCircleArc(pts[3], c, pts[0]),
        ]

        loop = geo.addCurveLoop(arcs)
        surfaces.append(geo.addPlaneSurface([loop]))

    geo.synchronize()

    # Aggiungi Physical Group per le superfici
    gmsh.model.addPhysicalGroup(2, surfaces, 1)
    gmsh.model.setPhysicalName(2, 1, "surfaces")

    gmsh.model.mesh.generate(2)
    folder = os.path.dirname(filename)
    os.makedirs(folder, exist_ok=True)
    gmsh.write(filename)
    gmsh.finalize()
    return
# }}}
# Select function {{{
def SelectFunction(func_name, module):
    # Get functions
    FUNCTION_REGISTRY = {
            name : func for name, func in inspect.getmembers(module, inspect.isfunction)
            }
    return FUNCTION_REGISTRY[func_name]
# }}}
# Select class {{{
def SelectClass(class_name, module):
    # Get classes
    CLASS_REGISTRY = {
            name : cls for name, cls in inspect.getmembers(module, inspect.isclass)
            }
    return CLASS_REGISTRY[class_name]
# }}}
# Plot polygons {{{
def PlotPolygons(polygons, filename):
    # Initialisation of points and polygons
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()
    pid = 0
    for poly in polygons:
        coords = np.asarray(poly.exterior.coords[:-1])
        numNods = len(coords)

        vtk_poly = vtk.vtkPolygon()
        vtk_poly.GetPointIds().SetNumberOfIds(numNods)

        for k1, (x, y) in enumerate(coords):
            points.InsertNextPoint(x, y, 0.0)
            vtk_poly.GetPointIds().SetId(k1, pid)
            pid += 1

        polys.InsertNextCell(vtk_poly)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()
    return
# }}}
