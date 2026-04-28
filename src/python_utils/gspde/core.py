# Libraries {{{
from mpi4py import MPI

import numpy as np

from shapely.geometry import Polygon
from shapely import centroid

from pdb import set_trace

from . import set_gspde, Dynamics
from ..mesh_utils import EquidistributeMesh
# }}}

# Class GSPDE {{{
class GSPDE(object):
    # Description {{{
    """
    GSPDE
    =====
    A general class for surface PDEs with mechanical and protrusive forces.

    Parameters
    ----------
    kwargs : dict
        Various keyword arguments controlling model setup:
        - model: gmsh model
        - quadrature_degree: quadrature degree
        - meshOrder: order of the mesh
        - aRef, periRef: reference area and perimeter
        - dt: time step
        - protrusion_function: custom force model (callable)
    """
    # }}}
    # Properties {{{
    @property
    def nodes(self):
        return self.domain.geometry.x[:, :self.dimSpa]
    @property
    def area(self):
        return self.GetPolygon().area
    @property
    def perimeter(self):
        return self.GetPolygon().length
    @property
    def centroid(self):
        if self.centroid_function is not None:
            return self.centroid_function()
        else:
            return np.array(centroid(self.GetPolygon()).coords[0])
    # }}}
    # __init__ {{{
    def __init__(self, kwargs):
        self.centroid_function = None
        self.kwargs = kwargs
        self.comm = kwargs["comm"]
        self.rank = self.comm.Get_rank()
        self.numRanks = self.comm.Get_size()
        self.Dynamics = kwargs.get("Dynamics", Dynamics.BasicDynamics)
        # Set domain
        set_gspde.set_domain(self)
        # Set measures
        set_gspde.set_measures(self)
        # Set finite element spaces
        set_gspde.set_fe_spaces(self)
        # Set dynamics: variables, expressions and initialisation
        self.Dynamics = self.Dynamics(self)
        # Set weak form
        self.Dynamics.set_weak_form()
        # Set nonlinear problem
        set_gspde.set_nonlinear_problem(self)
        return
    # }}}
    # Solve {{{
    def Solve(self):
        iters, converged = self.solver.solve(self.w)
        return iters, converged
    # }}}
    # Get polygon {{{
    def GetPolygon(self):
        orderedNodes = self.GetOrderedNodes()
        xCoor = orderedNodes[:-1, 0]
        yCoor = orderedNodes[:-1, 1]
        poly = Polygon(zip(xCoor, yCoor))
        return poly
    # }}}
    # Get ordered nodes {{{
    def GetOrderedNodes(self):
        # Get nodes
        nodes = self.domain.geometry.x[:, :self.dimSpa]
        # Get ordered nodes
        orderedNodes = nodes[self.orderedNodeIds]
        return orderedNodes
    # }}}
    # Equidistribute {{{
    def Equidistribute(self):
        orderedNodes = self.GetOrderedNodes()
        newOrderedNodes = EquidistributeMesh(orderedNodes, optimal = False)
        self.domain.geometry.x[self.orderedNodeIds, :self.dimSpa] = newOrderedNodes
        self.x_old.interpolate(self.x_expr_id)
        return
    # }}}
    # Write results {{{
    def WriteResults(self, t):
        self.output.WriteResults(t)
        return
    # }}}
    # Write report {{{
    def WriteReport(self, t):
        self.Dynamics.WriteReport(t)
        return
    # }}}
    # Solve iteration {{{
    def SolveIteration(self, t):
        self.Dynamics.SolveIteration(t)
        return
    # }}}
    # Close results {{{
    def CloseResults(self):
        self.output.Close()
        return
    # }}}
# }}}
