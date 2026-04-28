# Libraries {{{
import os, sys

from petsc4py import PETSc

from basix.ufl import element, mixed_element

import ufl
from ufl import grad, inner, derivative

from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import numpy as np
from pdb import set_trace

from .Dynamics import *
from ..mesh_utils import OrderNodeList
# }}}

# Set domain {{{
def set_domain(gspde):
    model_mesh = gspde.kwargs["model_mesh"]
    # Mesh
    gspde.domain, gspde.cellTags, gspde.facetTags = model_mesh
    # Get dimensions
    gspde.dimSur = gspde.domain.topology.dim
    gspde.dimSpa = gspde.dimSur + 1
    # Get number of nodes
    gspde.numNods, _ = gspde.domain.geometry.x.shape
    # Get ordered list of node ids
    connectivities = gspde.domain.geometry.dofmap
    gspde.numEles, nnod = connectivities.shape
    if nnod == 2:
        connectivities = np.column_stack([connectivities,
                                          -np.ones([gspde.numEles, 1], dtype = int)])
    gspde.orderedNodeIds = OrderNodeList(connectivities[0, 0],
                                         connectivities[0, 0],
                                         connectivities, gspde.numEles)
    return
# }}}

# Set measures {{{
def set_measures(gspde):
    kwargs = gspde.kwargs
    gspde.normalDirection = kwargs.get("normalDirection", 1.0)
    quadrature_degree = gspde.kwargs["quadrature_degree"]
    gspde.x = ufl.SpatialCoordinate(gspde.domain)
    gspde.n = ufl.CellNormal(gspde.domain)
    gspde.dx = ufl.Measure("dx", domain = gspde.domain,
                          metadata = {"quadrature_degree" : quadrature_degree,
                                      "quadrature_rule" : "default"})
    return
# }}}

# Set finite element spaces {{{
def set_fe_spaces(gspde):
    meshOrder = gspde.kwargs["meshOrder"]
    # Element types: quadratic scalar
    ele_scalar = element("Lagrange", gspde.domain.basix_cell(), meshOrder)
    gspde.V_scalar = fem.functionspace(gspde.domain, ele_scalar)
    # Element types: quadratic vector
    ele_u = element("Lagrange", gspde.domain.basix_cell(), meshOrder,
                    shape = (gspde.dimSpa, ))
    gspde.V_u = fem.functionspace(gspde.domain, ele_u)
    # Mixed element
    ele_mixed = mixed_element([ele_u, ele_scalar])
    gspde.V_mixed = fem.functionspace(gspde.domain, ele_mixed)
    return
# }}}

# Set nonlinear problem {{{
def set_nonlinear_problem(gspde):
    SetSolverOpt = gspde.kwargs["SetSolverOpt"]
    gspde.problem = NonlinearProblem(gspde.Res, gspde.w, [], gspde.tangent)
    gspde.solver = NewtonSolver(gspde.comm, gspde.problem)
    SetSolverOpt(gspde.solver)
    return
# }}}
