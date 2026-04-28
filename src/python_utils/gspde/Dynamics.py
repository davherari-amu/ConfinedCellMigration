# Libraries {{{
from petsc4py import PETSc
from dolfinx.fem import Constant, Function, Expression, form, assemble_scalar
import ufl
from ufl import grad, inner, derivative
from ufl import TestFunctions, TrialFunction, dot, div, grad

import numpy as np
from shapely import centroid
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

from pdb import set_trace
import os
import csv

from .curvature import InitialCurvature
from .forces import (SelfRepulsiveForce_kdtree, eaSumNodal,
                     ContactToECMPoints, ContactToECMPolygons)
from ..output_utils import Output
from ..misc_utils import FromVectorToMatrix, mprint
# }}}

# Set basic dynamics class {{{
class BasicDynamics(object):
    # __init__ {{{
    def __init__(self, gspde):
        self.gspde = gspde
        # Set basic variables
        self._set_basic_variables()
        # Set basic expressions
        self._set_basic_expressions()
        # Set basic initialisation
        self._set_basic_initialisation()
        # Add new variables, expressions and initialisations
        self.add_variables()
        self.add_expressions()
        self.add_initialisation()
        # Create total force expression
        gspde.barrierForce_expr = Expression(gspde.barrierForce_raw,
                                            gspde.V_scalar.element.interpolation_points())
        gspde.totalForce_expr = Expression(gspde.totalForce_raw,
                                            gspde.V_scalar.element.interpolation_points())
        # Set output
        self.set_output(gspde.kwargs["filename"])
        gspde.csvfile = os.path.join("results", gspde.kwargs["filename"] + ".csv")
        self.set_report()
        return
    # }}}
    # Set basic variables {{{
    def _set_basic_variables(self):
        gspde = self.gspde
        Href = gspde.kwargs["Href"]
        aRef = gspde.kwargs["aRef"]
        periRef = gspde.kwargs["periRef"]
        dt = gspde.kwargs["dt"]
        t = gspde.kwargs.get("t", 0.0)
        # Floats
        gspde.dt = dt
        gspde.aRef = aRef
        gspde.periRef = periRef
        gspde.Href = Href
        # Constants
        gspde.dk = Constant(gspde.domain, PETSc.ScalarType(dt))
        gspde.t_constant = Constant(gspde.domain, PETSc.ScalarType(t))
        # Main functions
        gspde.w = Function(gspde.V_mixed)
        gspde.u, gspde.H = ufl.split(gspde.w)
        gspde.u_test, gspde.H_test = TestFunctions(gspde.V_mixed)
        gspde.dw = TrialFunction(gspde.V_mixed)
        # Scalar functions
        gspde.H_old = Function(gspde.V_scalar)
        gspde.selfRepuForce = Function(gspde.V_scalar)
        gspde.bendingTensionRatio = Function(gspde.V_scalar)
        gspde.barrierForce = Function(gspde.V_scalar)
        gspde.totalForce = Function(gspde.V_scalar)
        gspde.arclen = Function(gspde.V_scalar)
        # Vector functions
        gspde.x_old = Function(gspde.V_u)
        gspde.x0 = Function(gspde.V_u)
        gspde.disp = Function(gspde.V_u)
        gspde.normal = Function(gspde.V_u)
        gspde.grad_H = Function(gspde.V_u)
        # Estimate reference radius
        r_ref = np.sqrt(gspde.aRef/np.pi)
        # List of output
        gspde.output_variables = [gspde.disp, gspde.H_old,
                                  gspde.selfRepuForce,
                                  gspde.barrierForce, gspde.totalForce,
                                  gspde.grad_H, gspde.normal,
                                  gspde.arclen]
        gspde.output_names = ["u", "H", "Fs", "Fb", "Ftot", "grad_H", "normal", "arclen"]
        # List of report variables
        gspde.report_names = ["time", "centroid_x", "centroid_y", "area",
                              "perimeter", "memb_tension", "memb_bending",
                              "force_self", "force_total", "grad_H",
                              "energy", "bending_energy"]
        energy_form = form((1.0 + 0.5*gspde.bendingTensionRatio*(gspde.H - 1.0)**2.0)*gspde.dx)
        bending_energy_form = form(((gspde.H - 1.0/r_ref)**2.0)*gspde.dx)
        memb_tension_form = form((-gspde.H)**2.0*gspde.dx)
        memb_bending_form = form((gspde.bendingTensionRatio*(div(grad(gspde.H)) + 0.5*gspde.H**3.0))**2.0*gspde.dx)
        force_self_form = form((gspde.selfRepuForce)**2.0*gspde.dx)
        force_total_form = form((gspde.totalForce)**2.0*gspde.dx)
        grad_H_form = form(dot(gspde.grad_H, gspde.grad_H)*gspde.dx)
        one = Constant(gspde.domain, PETSc.ScalarType(1.0))
        area_form = form(one*gspde.dx)
        gspde.report_list_func = lambda t : [t,
                                             centroid(gspde.GetPolygon()).coords[0][0],
                                             centroid(gspde.GetPolygon()).coords[0][1],
                                             gspde.area,
                                             gspde.perimeter,
                                             np.sqrt(assemble_scalar(memb_tension_form)),
                                             np.sqrt(assemble_scalar(memb_bending_form)),
                                             np.sqrt(assemble_scalar(force_self_form)),
                                             np.sqrt(assemble_scalar(force_total_form)),
                                             np.sqrt(assemble_scalar(grad_H_form)/assemble_scalar(area_form)),
                                             assemble_scalar(energy_form),
                                             assemble_scalar(bending_energy_form),
                                             ]
        return
    # }}}
    # Set basic expressions {{{
    def _set_basic_expressions(self):
        gspde = self.gspde
        normalDirection = gspde.normalDirection
        gspde.normal_expr = Expression(gspde.n, gspde.V_u.element.interpolation_points())
        gspde.x_expr_id = Expression(gspde.x, gspde.V_u.element.interpolation_points())
        gspde.x_expr = Expression(gspde.w.sub(0), gspde.V_u.element.interpolation_points())
        gspde.H_expr = Expression(gspde.w.sub(1), gspde.V_scalar.element.interpolation_points())
        gspde.grad_H_expr = Expression(grad(gspde.H) - gspde.normal*dot(gspde.normal, grad(gspde.H)), gspde.V_u.element.interpolation_points())
        gspde.disp_expr = Expression(gspde.x_old - gspde.x0, gspde.V_u.element.interpolation_points())
        gspde.totalForce_raw = gspde.selfRepuForce - normalDirection*gspde.barrierForce
        gspde.barrierForce_raw = Constant(gspde.domain, PETSc.ScalarType(0.0))
        return
    # }}}
    # Set basic initialisation {{{
    def _set_basic_initialisation(self):
        gspde = self.gspde
        bendingTensionRatio = gspde.kwargs.get("bendingTensionRatio", 1.0e-6)
        # Initial normal vector
        gspde.normal.interpolate(gspde.normal_expr)
        # Initial x (identity map)
        gspde.x_old.interpolate(gspde.x_expr_id)
        gspde.x0.interpolate(gspde.x_expr_id)
        # Initial curvature
        InitialCurvature(gspde.H_old, gspde.normal, gspde.x_old, gspde.dx,
                         comm = gspde.comm)
        # Bending tension ratio
        gspde.bendingTensionRatio.x.array[:] = bendingTensionRatio
        return
    # }}}
    # Add variables {{{
    def add_variables(self):
        return
    # }}}
    # Add expressions {{{
    def add_expressions(self):
        return
    # }}}
    # Add initialisation {{{
    def add_initialisation(self):
        return
    # }}}
    # Set weak form of velocity {{{
    def set_velocity_weak_form(self):
        gspde = self.gspde
        Href = gspde.Href
        B = gspde.bendingTensionRatio
        Mu = (1.0/gspde.dk)*inner(inner(gspde.u - gspde.x_old, gspde.normal),
                                  gspde.H_test)
        Su = inner(grad(gspde.H), grad(gspde.H_test))
        Hpow2 = gspde.H**2.0 - Href**2.0
        Qu = 0.5*inner(Hpow2*gspde.H, gspde.H_test)
        Tu = inner(gspde.H, gspde.H_test)
        Fu = inner(gspde.totalForce, gspde.H_test)
        Res_u = (Mu + Tu + B*(Su - Qu) - Fu)*gspde.dx
        return Res_u
    # }}}
    # Set weak form of curvature {{{
    def set_curvature_weak_form(self):
        gspde = self.gspde
        MH = inner(gspde.H*gspde.normal, gspde.u_test)
        SH = inner(grad(gspde.u), grad(gspde.u_test))
        Res_H = (MH - SH)*gspde.dx # H = div(x)
        return Res_H
    # }}}
    # Set weak form {{{
    def set_weak_form(self):
        gspde = self.gspde
        Res_u = self.set_velocity_weak_form()
        Res_H = self.set_curvature_weak_form()
        gspde.Res = Res_u + Res_H
        gspde.tangent = derivative(gspde.Res, gspde.w, gspde.dw)
        return
    # }}}
    # Set output {{{
    def set_output(self, filename):
        gspde = self.gspde
        gspde.output= Output(gspde.domain, gspde.output_variables,
                gspde.output_names,
                filename, gspde.comm)
        return
    # }}}
    # Set report {{{
    def set_report(self):
        gspde = self.gspde
        with open(gspde.csvfile, 'w', newline = "") as fle:
            writer = csv.writer(fle)
            writer.writerow(gspde.report_names)
        return
    # }}}
    # Write report {{{
    def WriteReport(self, time):
        gspde = self.gspde
        report_list = gspde.report_list_func(time)
        with open(gspde.csvfile, 'a', newline = "") as fle:
            writer = csv.writer(fle)
            writer.writerow(report_list)
        return
    # }}}
    # Self-repulsive force {{{
    def SelfRepulsiveForce(self):
        gspde = self.gspde
        tol = gspde.kwargs["self_tol"]
        stiffness  = gspde.kwargs["self_stiffness"]
        normalArray = FromVectorToMatrix(gspde.normal.x.array, gspde.dimSpa)
        xArray = gspde.domain.geometry.x[:, :gspde.dimSpa]
        repuForce = SelfRepulsiveForce_kdtree(xArray, normalArray,
                                              tol, stiffness)
        gspde.selfRepuForce.x.array[:] = repuForce
        return
    # }}}
    # Update loads {{{
    def UpdateLoads(self):
        gspde = self.gspde
        # Self-repulsive force
        self.SelfRepulsiveForce()
        return
    # }}}
    # Update variables {{{
    def UpdateVariables(self):
        gspde = self.gspde
        equidistribute = gspde.kwargs.get("equidistribute", True)
        # Update current position and curvature
        gspde.x_old.interpolate(gspde.x_expr)
        gspde.H_old.interpolate(gspde.H_expr)
        gspde.grad_H.interpolate(gspde.grad_H_expr)
        # Update mesh
        uMat = FromVectorToMatrix(gspde.x_old.x.array, gspde.dimSpa)
        gspde.domain.geometry.x[:, :gspde.dimSpa] = uMat
        # Mesh tangential movement for equidistribution
        if equidistribute:
            gspde.Equidistribute()
        # Update total displacement
        gspde.disp.interpolate(gspde.disp_expr)
        # Update normal
        gspde.normal.interpolate(gspde.normal_expr)
        # Update arclen
        delta_arclen = gspde.perimeter/gspde.numNods
        arclen = np.arange(gspde.numNods)*delta_arclen
        gspde.arclen.x.array[gspde.orderedNodeIds[:-1]] = arclen
        return
    # }}}
    # Solve {{{
    def SolveIteration(self, t):
        gspde = self.gspde
        # Update time
        gspde.t_constant.value = t
        # Update loads
        self.UpdateLoads()
        # Interpolate barrier force
        gspde.barrierForce.interpolate(gspde.barrierForce_expr)
        # Interpolate total force
        gspde.totalForce.interpolate(gspde.totalForce_expr)
        # Solve
        gspde.Solve()
        gspde.w.x.scatter_forward()
        # Update variables
        self.UpdateVariables()
        return
    # }}}
# }}}

# Size conservation {{{
class SizeControlDynamics(BasicDynamics):
    # Add variables {{{
    def add_variables(self):
        super().add_variables()
        gspde = self.gspde
        kwargs = gspde.kwargs
        # Floats
        gspde.area_stiffness = kwargs["area_stiffness"]
        gspde.peri_stiffness = kwargs["peri_stiffness"]
        gspde.peri_max_factor = kwargs.get("peri_max_factor", 1.0)
        # Constants
        gspde.opre_area = Constant(gspde.domain, PETSc.ScalarType(0.0))
        gspde.opre_peri = Constant(gspde.domain, PETSc.ScalarType(0.0))
        # Scalar functions
        gspde.opre = Function(gspde.V_scalar)
        gspde.smoothed_H = Function(gspde.V_scalar)
        # List of output
        gspde.output_variables += [gspde.opre, gspde.smoothed_H]
        gspde.output_names += ["opre", "smooth_H"]
        # List to report
        force_osmo_form = form((gspde.opre)**2.0*gspde.dx)
        gspde.report_names += ["force_osmo"]
        old_func = gspde.report_list_func
        gspde.report_list_func = lambda t : old_func(t) + [np.sqrt(assemble_scalar(force_osmo_form))]
        return
    # }}}
    # Add expressions {{{
    def add_expressions(self):
        super().add_expressions()
        gspde = self.gspde
        # Modify total force
        op = gspde.opre_peri
        sigmoid = lambda x: 1.0/(1.0 + ufl.exp(-x))
        # opre_H = sigmoid(10.0*(gspde.smoothed_H - gspde.Href))*sigmoid(-10.0*op)*gspde.smoothed_H*op
        opre_H = sigmoid(-10.0*op)*gspde.smoothed_H*op
        opre_total = (opre_H + gspde.opre_area)
        gspde.opre_expr = Expression(opre_total, gspde.V_scalar.element.interpolation_points())
        gspde.totalForce_raw += gspde.opre
        return
    # }}}
    # Osmotic pressure {{{
    def OsmoticPressure(self):
        gspde = self.gspde
        self.SmoothCurvature()
        area_stiffness = gspde.area_stiffness
        peri_stiffness = gspde.peri_stiffness
        max_peri = gspde.peri_max_factor*gspde.periRef
        # lambda_peri = (-1.0 - peri_stiffness*(gspde.perimeter - gspde.periRef))
        lambda_peri = peri_stiffness*(max_peri - gspde.perimeter)
        lambda_area = area_stiffness*(gspde.aRef - gspde.area)
        gspde.opre_peri.value = lambda_peri
        gspde.opre_area.value = lambda_area
        return
    # }}}
    # Update loads {{{
    def UpdateLoads(self):
        super().UpdateLoads()
        gspde = self.gspde
        # Osmotic pressure
        self.OsmoticPressure()
        gspde.opre.interpolate(gspde.opre_expr)
        return
    # }}}
    # Compute smoothed curvature {{{
    def SmoothCurvature(self):
        gspde = self.gspde
        # Smoothing parameters
        window_length = 5  # Must be odd
        polyorder = 2       # Order of polynomial
        orderedArray = gspde.H_old.x.array[gspde.orderedNodeIds]
        # smoothed_array = savgol_filter(orderedArray, window_length, polyorder)
        smoothed_array = gaussian_filter1d(orderedArray, 10)
        gspde.smoothed_H.x.array[gspde.orderedNodeIds] = smoothed_array
        return
    # }}}
# }}}

# Random filopodia dynamics {{{
class RandomFilopodiaDynamics(BasicDynamics):
    # Add variables {{{
    def add_variables(self):
        super().add_variables()
        gspde = self.gspde
        kwargs = gspde.kwargs
        # Scalar functions
        gspde.ea = Function(gspde.V_scalar)
        gspde.distance_from_centroid = Function(gspde.V_scalar)
        # Vector functions
        gspde.filoDir = Function(gspde.V_u)
        # List of output
        gspde.output_variables += [gspde.ea, gspde.filoDir, gspde.distance_from_centroid]
        gspde.output_names += ["ea", "filoDir", "distance_from_centroid"]
        # Functions
        ea_params = gspde.kwargs["ea_params"]
        gspde.eaFunc = eaSumNodal(**ea_params)
        gspde.ea_intersection = ea_params.get("ea_intersection", "sum")
        # Function options
        gspde.protrusion_direction = ea_params.get("direction", "source")
        gspde.max_peri_factor = ea_params.get("max_peri_factor", None)
        # List to report
        force_ea_form = form((gspde.ea)**2.0*gspde.dx)
        gspde.report_names += ["force_ea", "num_filo_peaks", "length_filos"]
        old_func = gspde.report_list_func
        gspde.report_list_func = lambda t : (old_func(t) + [np.sqrt(assemble_scalar(force_ea_form))] +
                                                           self.MeasurePeaks())
        return
    # }}}
    # Add expressions {{{
    def add_expressions(self):
        super().add_expressions()
        gspde = self.gspde
        # Modify total force
        gspde.totalForce_raw += gspde.ea*dot(gspde.filoDir, gspde.normal)
        return
    # }}}
    # Update loads {{{
    def UpdateLoads(self):
        super().UpdateLoads()
        gspde = self.gspde
        # Filopodia force
        self.FilopodiaForce()
        # Penalise force by maximum perimeter
        if not gspde.max_peri_factor is None:
            maxPeri = gspde.max_peri_factor*gspde.periRef
            pena_peri = 1.0/(1.0 + np.exp(10.0*(gspde.perimeter - maxPeri)))
            gspde.ea.x.array[:] *= pena_peri
        # Penalise ea by memb_memb_force and adhesion_cell_cell
        ea_params = gspde.kwargs["ea_params"]
        memb_memb_pena = ea_params.get("memb_memb_pena", 1.0)
        if hasattr(gspde, 'memb_memb_force'):
            penalise_at = np.abs(gspde.memb_memb_force.x.array) > 0.0
            gspde.ea.x.array[penalise_at] *= memb_memb_pena
        if hasattr(gspde, 'adhesion_cell_cell'):
            penalise_at = np.abs(gspde.adhesion_cell_cell.x.array) > 0.0
            gspde.ea.x.array[penalise_at] *= memb_memb_pena
        return
    # }}}
    # Filopodia force {{{
    def FilopodiaForce(self):
        gspde = self.gspde
        meshOrder = gspde.kwargs["meshOrder"]
        ea_params = gspde.kwargs["ea_params"]
        typ = ea_params.get("typ", "pressure")
        maxLen = ea_params.get("maxLen", None)
        # Initialisation
        gspde.ea.x.array[:] = 0.0
        eaFunc = gspde.eaFunc
        # Global force
        nodes = gspde.domain.geometry.x[:, :gspde.dimSpa]
        normalArray = FromVectorToMatrix(gspde.normal.x.array, gspde.dimSpa)
        indices, forces, directions = eaFunc(nodes, gspde.t_constant,
                                             h = gspde.perimeter/(gspde.numEles*meshOrder),
                                             centroid = gspde.centroid,
                                             typ = typ,
                                             maxLen = maxLen,
                                             normal = normalArray)
        # Local force
        for k2 in range(len(indices)):
            index = indices[k2]
            force = forces[k2]
            direction = directions[k2]
            # Assign values
            if gspde.ea_intersection == "sum":
                gspde.ea.x.array[index] += force
            elif gspde.ea_intersection == "max":
                ea_index = gspde.ea.x.array[index]
                gspde.ea.x.array[index] = np.max(np.column_stack([ea_index, force]), axis = 1)
            else:
                raise ValueError("Invalid ea_intersection, use 'sum' or 'max'")
            gspde.filoDir.x.array[index*2] = direction[:, 0]
            gspde.filoDir.x.array[index*2 + 1] = direction[:, 1]
        if gspde.protrusion_direction == "normal":
            gspde.filoDir.x.array[:] = gspde.normal.x.array[:]
        elif gspde.protrusion_direction == "source":
            pass
        else:
            raise ValueError("Wrong protrusion_direction.")
        return
    # }}}
    # Update variables {{{
    def UpdateVariables(self):
        super().UpdateVariables()
        gspde = self.gspde
        centroid = gspde.centroid
        nodes = gspde.nodes
        # Distances from centroid to nodes
        distances = np.linalg.norm(nodes - centroid, axis = 1)
        gspde.distance_from_centroid.x.array[:] = distances[:]
        return
    # }}}
    # Measure peaks {{{
    def MeasurePeaks(self, prominence_factor = 0.05):
        gspde = self.gspde
        # Find peaks
        orderedArray = gspde.distance_from_centroid.x.array[gspde.orderedNodeIds]
        peaks, props = find_peaks(orderedArray,
                                  prominence = prominence_factor*orderedArray.max())
        numPeaks = peaks.shape[0]
        lenPeaks = props["prominences"]
        return [numPeaks, lenPeaks]
    # }}}
# }}}

# Plasma-membrane + nuclear envelope {{{
class MembraneNucleusDynamics(BasicDynamics):
    # Add variables {{{
    def add_variables(self):
        super().add_variables()
        gspde = self.gspde
        kwargs = gspde.kwargs
        # Scalar functions
        gspde.memb_nuen_force = Function(gspde.V_scalar)
        gspde.memb_nuen_spring = Function(gspde.V_scalar)
        gspde.dot_dis_nor = Function(gspde.V_scalar)
        # Constants
        gspde.memb_nuen_distance = Constant(gspde.domain,
                                            np.array([0.0, 0.0],
                                            dtype = np.float64))
        spring_stiffness = kwargs["spring_stiffness"]
        gspde.spring_stiffness = Constant(gspde.domain, spring_stiffness)
        # List of output
        gspde.output_variables += [gspde.memb_nuen_force, gspde.memb_nuen_spring]
        gspde.output_names += ["memb_nuen_force", "memb_nuen_spring"]
        # List to report
        force_memb_nuen_form = form((gspde.memb_nuen_force)**2.0*gspde.dx)
        force_memb_nuen_spring_form = form((gspde.memb_nuen_spring)**2.0*gspde.dx)
        gspde.report_names += ["force_memb_nuen", "force_memb_nuen_spring"]
        old_func = gspde.report_list_func
        gspde.report_list_func = lambda t : old_func(t) + [np.sqrt(assemble_scalar(force_memb_nuen_form)),
                                                           np.sqrt(assemble_scalar(force_memb_nuen_spring_form))]
        return
    # }}}
    # Add expressions {{{
    def add_expressions(self):
        super().add_expressions()
        gspde = self.gspde
        # Forms
        gspde.dot_dis_nor_form = form(gspde.dot_dis_nor*gspde.dx)
        # Expressions
        gspde.dot_dis_nor_expr = Expression(dot(gspde.memb_nuen_distance, gspde.normal),
                                            gspde.V_scalar.element.interpolation_points())
        # Modify total force
        gspde.barrierForce_raw += gspde.memb_nuen_force
        gspde.totalForce_raw += gspde.memb_nuen_spring
        return
    # }}}
# }}}

# Nucleus dynamics with respect to plasma-membrane {{{
class NucleusToPMDynamics(MembraneNucleusDynamics):
    # Add variables {{{
    def add_variables(self):
        super().add_variables()
        gspde = self.gspde
        kwargs = gspde.kwargs
        # Constants
        gspde.viscosityRatio = Constant(gspde.domain, kwargs["viscosityRatio"])
        gspde.tensionRatio = Constant(gspde.domain, kwargs["tensionRatio"])
        gspde.bendingRatio = Constant(gspde.domain, kwargs["bendingRatio"])
        return
    # }}}
    # Set weak form of velocity {{{
    def set_velocity_weak_form(self):
        gspde = self.gspde
        Href = gspde.Href
        B = gspde.bendingTensionRatio
        Rvisco = gspde.viscosityRatio
        Rtension = gspde.tensionRatio
        Rbending = gspde.bendingRatio
        Mu = (Rtension/Rvisco)*(1.0/gspde.dk)*inner(inner(gspde.u - gspde.x_old, gspde.normal),
                                  gspde.H_test)
        Su = inner(grad(gspde.H), grad(gspde.H_test))
        Hpow2 = gspde.H**2.0 - Href**2.0
        Qu = 0.5*inner(Hpow2*gspde.H, gspde.H_test)
        Tu = inner(gspde.H, gspde.H_test)
        Fu = inner(gspde.totalForce, gspde.H_test)
        Res_u = (Mu + Tu + (Rtension/Rbending)*B*(Su - Qu) - Rtension*Fu)*gspde.dx
        return Res_u
    # }}}
# }}}

# Extracellular matrix barrier dynamics {{{
# Base dynamics {{{
class ECMBarrierDynamics(BasicDynamics):
    # Add variables {{{
    def add_variables(self):
        super().add_variables()
        gspde = self.gspde
        kwargs = gspde.kwargs
        ecm_params = kwargs["ecm_params"]
        # Scalar functions
        gspde.ecm_force = Function(gspde.V_scalar)
        # List of output
        gspde.output_variables += [gspde.ecm_force]
        gspde.output_names += ["ecm_force"]
        # Functions
        self.ecm_function = ContactToECMPoints(**ecm_params)
        # List to report
        force_ecm_form = form((gspde.ecm_force)**2.0*gspde.dx)
        gspde.report_names += ["force_ecm"]
        old_func = gspde.report_list_func
        gspde.report_list_func = lambda t : old_func(t) + [np.sqrt(assemble_scalar(force_ecm_form))]
        return
    # }}}
    # Add expressions {{{
    def add_expressions(self):
        super().add_expressions()
        gspde = self.gspde
        # Modify total force
        gspde.barrierForce_raw += gspde.ecm_force
        return
    # }}}
    # Update loads {{{
    def UpdateLoads(self):
        super().UpdateLoads()
        gspde = self.gspde
        # ECM force
        self.Update_ecm_force()
        return
    # }}}
    # Update extracellular matrix point force on surface {{{
    def Update_ecm_force(self):
        ecm_function = self.ecm_function
        gspde = self.gspde
        force = ecm_function(self.gspde)
        gspde.ecm_force.x.array[:] = force[:]
        return
    # }}}
# }}}
# Extracellular matrix barrier point dynamics {{{
class ECMBarrierPointDynamics(ECMBarrierDynamics):
    # Add variables {{{
    def add_variables(self):
        super().add_variables()
        gspde = self.gspde
        kwargs = gspde.kwargs
        ecm_params = kwargs["ecm_params"]
        # Functions
        self.ecm_function = ContactToECMPoints(**ecm_params)
        return
    # }}}
# }}}
# Extracellular matrix barrier polygon dynamics {{{
class ECMBarrierPolygonDynamics(ECMBarrierDynamics):
    # Add variables {{{
    def add_variables(self):
        super().add_variables()
        gspde = self.gspde
        kwargs = gspde.kwargs
        ecm_params = kwargs["ecm_params"]
        # Functions
        self.ecm_function = ContactToECMPolygons(**ecm_params)
        return
    # }}}
# }}}
# }}}

# External pressure gradient {{{
class ExternalPressureGradient(BasicDynamics):
    # Add variables {{{
    def add_variables(self):
        super().add_variables()
        gspde = self.gspde
        kwargs = gspde.kwargs
        # Scalar functions
        gspde.pressureGradient = Function(gspde.V_scalar)
        # Function
        gspde.pressure_func = kwargs["pressure_func"]
        # List of output
        gspde.output_variables += [gspde.pressureGradient]
        gspde.output_names += ["pressure_gradient"]
        # List to report
        pressureGradient_form = form((gspde.pressureGradient)**2.0*gspde.dx)
        gspde.report_names += ["pressure_gradient"]
        old_func = gspde.report_list_func
        gspde.report_list_func = lambda t : old_func(t) + [np.sqrt(assemble_scalar(pressureGradient_form))]
    # }}}
    # Add expressions {{{
    def add_expressions(self):
        super().add_expressions()
        gspde = self.gspde
        # Modify total force
        gspde.totalForce_raw += gspde.pressureGradient
        return
    # }}}
    # Update loads {{{
    def UpdateLoads(self):
        super().UpdateLoads()
        gspde = self.gspde
        # Pressure gradient
        pressure = gspde.pressure_func(gspde)
        gspde.pressureGradient.x.array[:] = pressure[:]
        return
    # }}}
# }}}
