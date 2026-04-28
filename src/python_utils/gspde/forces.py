# Libraries {{{
from dolfinx import log
from dolfinx.fem import assemble_scalar

from scipy.spatial import KDTree
import numpy as np
import shapely
from shapely.geometry import Point
from shapely import centroid

from abc import ABC, abstractmethod
import copy

from pdb import set_trace

from mpi4py import MPI
from ..misc_utils import FromVectorToMatrix, mprint
from ..mesh_utils import ChebyshevCenter
# }}}

# Self repulsive force using KDTree {{{
def SelfRepulsiveForce_kdtree(xArray, normalArray, tol, stiffness,
                              normal_tol = -0.7, tol_factor = 2.0,
                              normal_st = 20.0):
    numPoints, _ = xArray.shape
    # Set up KDTree
    tree = KDTree(xArray)
    # Find indices of closest points
    closest_point_ids = tree.query_ball_point(xArray, tol*tol_factor)
    # Compute repulsive force from distance and dot normals
    repuForce = np.zeros(numPoints)
    for k1 in range(numPoints):
        # Get point and normal and ids of closest points
        point = xArray[k1]
        normal = normalArray[k1]
        ids = closest_point_ids[k1]
        # Compute distance from point to point
        distance_vector = xArray[ids] - point
        distance_point = np.linalg.norm(distance_vector, axis = 1)
        distance = distance_point - tol
        negative_ids = distance < 0.0
        distance_contribution = np.zeros_like(distance)
        distance_contribution[negative_ids] = -distance[negative_ids]*stiffness
        # Compute normal contribution
        normals = normalArray[ids]
        dots = np.dot(normals, normal)
        normal_contribution = (np.tanh(-normal_st*(dots - normal_tol)) + 1.0)/2.0
        # Repulsive sign
        sign_dots = np.ones_like(distance_contribution)
        neg_force_arg = np.argwhere(np.dot(distance_vector, normal) > 0)
        sign_dots[neg_force_arg] = -1.0
        # Compute force
        full_contribution = normal_contribution*distance_contribution*sign_dots
        # Maximum arg
        argmax = np.argmax(np.abs(full_contribution))
        repuForce[k1] = full_contribution[argmax]
    return repuForce
# }}}

# Protruding forces {{{
# Class to sum eas defined by nodes {{{
class eaSumNodal(object):
    # Properties {{{
    @property
    def eas(self):
        return self._eas
    @property
    def N(self):
        return self._N
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._N = kwargs["N"]
        hs = kwargs["hs"]
        Ts = kwargs["Ts"]
        t0s = kwargs["t0s"]
        thetas = kwargs["thetas"]
        ws = kwargs["ws"]
        ForceClass = kwargs["ForceClass"]
        startAt = kwargs.get("startAt", 0.0)
        sourceDia = kwargs.get("sourceDia", 1.0)
        # New kwargs
        keys_to_remove = {"N", "hs", "Ts", "t0s", "thetas", "ws",
                          "ForceClass", "startAt", "sourceDia"}
        new_kwargs = {key : arg for key, arg in kwargs.items() if key not in keys_to_remove}
        # Create random parameters
        t0s = t0s - t0s.min() + startAt
        # Create eas
        eas = [None]*self.N
        for k1 in range(self.N):
            theta = thetas[k1]
            source = np.array([np.cos(theta), np.sin(theta)])*sourceDia
            eas[k1] = ForceClass(source = source,
                                 t0 = t0s[k1],
                                 width = ws[k1],
                                 period = Ts[k1],
                                 magnitude = hs[k1],
                                 **new_kwargs)
        self._eas = eas
        return
    # }}}
    # __call__ {{{
    def __call__(self, xArray, t, **kwargs):
        indices = []
        forces = []
        directions = []
        for ea_k1 in self.eas:
            if ea_k1.t0 <= t.value and t.value <= ea_k1.t0 + ea_k1.period:
                indices_k1, force_k1, direction_k1 = ea_k1(xArray, t, **kwargs)
                indices.append(indices_k1)
                # forces.append(np.ones(indices_k1.size)*force_k1)
                forces.append(force_k1)
                directions.append(direction_k1)
        return indices, forces, directions
    # }}}
# }}}
# ea_Dotpro: Class to define ea by dot product with source direction {{{
class ea_Dotpro(object):
    # Properties {{{
    @property
    def magnitude(self):
        return self._magnitude
    @property
    def width(self):
        return self._width
    @property
    def source(self):
        return self._source
    @property
    def t0(self):
        return self._t0
    @property
    def period(self):
        return self._period
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._magnitude = kwargs["magnitude"]
        self._width = kwargs["width"]
        self._source = kwargs["source"]
        self._t0 = kwargs["t0"]
        self._period = kwargs["period"]
        return
    # }}}
    # Active state {{{
    def active(self, time):
        return time >= self.t0 and time < self.t0 + self.period
    # }}}
    # Space functions {{{
    def eaSpaceFunc(self, xArray, h, centroid = np.zeros(2),
                    typ = "force", maxLen = None, **kwargs):
        movedArray = xArray - centroid
        # Define the number of points and effective force
        numForcePoints = np.ceil(self.width/h)
        if typ == "pressure":
            effForce = self.magnitude
        elif typ == "force":
            effForce = self.magnitude*h/self.width
        else:
            raise TypeError("Incorrect 'typ', use 'pressure' or 'force'")
        # Find the point which is best aligned with the source direction
        norm_x = np.linalg.norm(movedArray, axis = 1)
        unit_x = movedArray/norm_x[:, None]
        dots = np.dot(unit_x, self.source)
        norm_x_at_max_dots = np.where(dots > dots.max()*0.999, norm_x, 0.0)
        filCentre = movedArray[np.argmax(norm_x_at_max_dots)]
        node_id = np.argmax(dots)
        filCentre = movedArray[node_id]
        # Check if filopodia are too long
        if not maxLen is None:
            # Compute length from geometry centre
            filLength = np.linalg.norm(filCentre)
            # Reduce force if necessary
            lenDiff = filLength - maxLen
            if lenDiff >= 0.0:
                effForce = np.exp(-lenDiff/(0.1*maxLen))*effForce
        # Find the nearest using KDTree
        tree = KDTree(movedArray)
        distance, indices = tree.query(filCentre, k = numForcePoints)
        # Find direction
        if isinstance(indices, np.int64):
            direction = np.zeros_like(xArray[[indices]])
        else:
            direction = np.zeros_like(xArray[indices])
        direction[:, 0] = self.source[0]
        direction[:, 1] = self.source[1]
        # Compute force by distance
        minDis = distance.min()
        maxDis = distance.max()
        normalisedDis = (1.0/(minDis - maxDis))*(distance - maxDis)
        sigmoid = lambda x, steepness, x0: 1.0/(1.0 + np.exp(-steepness*(x - x0)))
        effForce = sigmoid(normalisedDis, 50.0, 0.1)*effForce
        return indices, effForce, direction
    # }}}
    # Time function {{{
    def eaTimeFunc(self, t):
        # Compute value
        if self.active(t.value):
            return 1.0
        else:
            return 0.0
    # }}}
    # __call__ {{{
    def __call__(self, xArray, t, **kwargs):
        indices = []
        forces = []
        directions = []
        eaTime = 0.0
        if self.active(t.value):
            eaTime = self.eaTimeFunc(t)
            indices, effForce, direction = self.eaSpaceFunc(xArray, **kwargs)
        return indices, effForce*eaTime, direction
    # }}}
# }}}
# }}}

# Contact inner-outer surfaces {{{
def ContactInnerOuterSurfaces(inner, outer, stiffness, tol = 0.0):
    # Description {{{
    """
    Compute the contact forces between two surfaces where one is inside
    the other.
    """
    # }}}
    # Get polygons
    poly_inner = inner.GetPolygon()
    poly_outer = outer.GetPolygon()
    # Initialisation of contact force
    contForce_inner = np.zeros(inner.numNods)
    contForce_outer = np.zeros(outer.numNods)
    # Check if inner is inside outer
    signDist_inner = np.zeros(inner.numNods)
    signDist_outer = np.zeros(outer.numNods)
    # Get surface points
    xArray_inner = inner.domain.geometry.x[:, :inner.dimSpa]
    xArray_outer = outer.domain.geometry.x[:, :outer.dimSpa]
    # Get points of inner outside outer
    for k1 in range(inner.numNods):
        pt = Point(xArray_inner[k1])
        inside = poly_outer.contains_properly(pt)
        dist = poly_outer.exterior.distance(pt)
        if inside:
            signDist_inner[k1] = dist - tol
        else:
            signDist_inner[k1] = -dist - tol
    inner_ids = signDist_inner < 0.0
    contForce_inner[inner_ids] = -signDist_inner[inner_ids]*stiffness
    # Get points of outer inside inner
    for k1 in range(outer.numNods):
        pt = Point(xArray_outer[k1])
        inside = poly_inner.contains_properly(pt)
        dist = poly_inner.exterior.distance(pt)
        if inside:
            signDist_outer[k1] = dist + tol
        else:
            signDist_outer[k1] = -dist + tol
    outer_ids = signDist_outer > 0.0
    contForce_outer[outer_ids] = -signDist_outer[outer_ids]*stiffness
    return contForce_inner, contForce_outer
# }}}

# Contact outer-outer surfaces {{{
def ContactOuterOuterSurfaces(surf1, surf2, stiffness, tol = 0.0):
    # Description {{{
    """
    Compute the contact forces between two surfaces not intersecting one
    another.
    """
    # }}}
    # Get polygons
    poly_surf1 = surf1.GetPolygon()
    poly_surf2 = surf2.GetPolygon()
    # Initialisation of contact force
    contForce_surf1 = np.zeros(surf1.numNods)
    contForce_surf2 = np.zeros(surf2.numNods)
    # Initialisation of signed distance
    signDist_surf1 = np.zeros(surf1.numNods)
    signDist_surf2 = np.zeros(surf2.numNods)
    # Get surface points
    xArray_surf1 = surf1.domain.geometry.x[:, :surf1.dimSpa]
    xArray_surf2 = surf2.domain.geometry.x[:, :surf2.dimSpa]
    # Get points of surf1 inside surf2
    for k1 in range(surf1.numNods):
        pt = Point(xArray_surf1[k1])
        inside = poly_surf2.contains_properly(pt)
        dist = poly_surf2.exterior.distance(pt)
        if inside:
            signDist_surf1[k1] = dist + tol
        else:
            signDist_surf1[k1] = -dist + tol
    surf1_ids = signDist_surf1 > 0.0
    contForce_surf1[surf1_ids] = signDist_surf1[surf1_ids]*stiffness
    # Get points of surf2 inside surf1
    for k1 in range(surf2.numNods):
        pt = Point(xArray_surf2[k1])
        inside = poly_surf1.contains_properly(pt)
        dist = poly_surf1.exterior.distance(pt)
        if inside:
            signDist_surf2[k1] = dist + tol
        else:
            signDist_surf2[k1] = -dist + tol
    surf2_ids = signDist_surf2 > 0.0
    contForce_surf2[surf2_ids] = signDist_surf2[surf2_ids]*stiffness
    return contForce_surf1, contForce_surf2
# }}}

# Contact point array with a closed surface {{{
def ContactWithClosedSurface(xArray_surf1, poly_surf2, stiffness, tol = 0.0):
    # Description {{{
    """
    Compute the contact forces on a point array (xArray_surf1) due to
    the presence of a closed curve (poly_surf2).
    """
    # }}}
    # Get number of nodes
    numNods_surf1 = xArray_surf1.shape[0]
    # Initialisation of contact force
    contForce_surf1 = np.zeros(numNods_surf1)
    # Get points of surf1 inside surf2
    signDist_surf1 = -SignDistance(xArray_surf1, poly_surf2) + tol
    surf1_ids = signDist_surf1 > 0.0
    contForce_surf1[surf1_ids] = signDist_surf1[surf1_ids]*stiffness
    return contForce_surf1
# }}}

# Sign distance with shapely {{{
def SignDistance(xArray, poly):
    pts = shapely.points(xArray)
    inside = shapely.contains_properly(poly, pts)
    dist = poly.exterior.distance(pts)
    return np.where(inside, -dist, dist)
# }}}

# Contact to extracellular matrix {{{
# Base {{{
class ContactToECM(object):
    # Properties {{{
    @property
    def stiffness(self):
        return self._stiffness
    @property
    def tol(self):
        return self._tol
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._tol = kwargs.get("ecm_tol")
        self._stiffness = kwargs["ecm_stiffness"]
        return
    # }}}
# }}}
# Contact to ecm points {{{
class ContactToECMPoints(ContactToECM):
    # Properties {{{
    @property
    def points(self):
        return self._points
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._points = kwargs["ecm_points"]
        return
    # }}}
    # __call__ {{{
    def __call__(self, gspde):
        points = self.points
        tol = self.tol
        stiffness = self.stiffness
        poly = gspde.GetPolygon()
        nodes = gspde.nodes
        # Get ecm points in contact
        signDist = SignDistance(points, poly)
        signDist_tol = signDist - tol
        points_in_contact_ids = np.where(signDist_tol < 0.0)[0]
        if points_in_contact_ids.shape[0] == 0:
            return np.zeros(gspde.numNods)
        points_in_contact = points[points_in_contact_ids]
        points_distance = signDist_tol[points_in_contact_ids]
        # Get distance at each point of the surface to each point of the ecm
        diff = points_in_contact[:, None, :] - nodes[None, :, :]
        distances = np.linalg.norm(diff, axis = 2)
        # Consider tolerance offset and keep the minimum
        distances = distances + points_distance[:, None] - tol
        distances = distances.min(axis = 0)
        # Keep only negative distances
        distances = np.where(distances < 0.0, distances, 0.0)
        # Define force
        force = distances*stiffness
        return -force
    # }}}
# }}}
# Contact to ecm polygons {{{
class ContactToECMPolygons(ContactToECM):
    # Properties {{{
    @property
    def polygons(self):
        return self._polygons
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._polygons = kwargs["ecm_points"]
        return
    # }}}
    # __call__ {{{
    def __call__(self, gspde):
        polygons = self.polygons
        tol = self.tol
        stiffness = self.stiffness
        nodes = gspde.nodes
        # Compute distance from nodes to polygons
        distances = [SignDistance(nodes, poly) - tol for poly in polygons]
        distances = np.min(np.column_stack(distances), axis = 1)
        # Keep only negative distances
        distances = np.where(distances < 0.0, distances, 0.0)
        # Define force
        force = distances*stiffness
        return -force
    # }}}
# }}}
# }}}

# Spring force from surface to surface centroid {{{
def SpringForceCentroidToCentroid(inner, outer):
    poly_inner = inner.GetPolygon()
    poly_outer = outer.GetPolygon()
    # Get centres
    inner_centre = np.array(centroid(poly_inner).coords[0])
    outer_centre = np.array(centroid(poly_outer).coords[0])
    # outer_centre = ChebyshevCenter(poly_outer, tol = 1.0e-3)
    # Get distance vector
    distance = outer_centre - inner_centre
    inner.memb_nuen_distance.value[:] = distance[:]
    outer.memb_nuen_distance.value[:] = -distance[:]
    # Total force
    dis_mag = np.sqrt(distance[0]**2.0 + distance[1]**2.0)
    total_force = dis_mag*inner.spring_stiffness.value
    # Interpolate functions
    inner.dot_dis_nor.interpolate(inner.dot_dis_nor_expr)
    outer.dot_dis_nor.interpolate(outer.dot_dis_nor_expr)
    # # nucleus pushes and cell pulls, compute active regions
    # inner.dot_dis_nor.x.array[inner.dot_dis_nor.x.array < 0.0] = 0.0
    # outer.dot_dis_nor.x.array[outer.dot_dis_nor.x.array > 0.0] = 0.0
    # Force proportional to projected distance
    inner_centre_to_outer_nodes = outer.nodes - inner_centre
    outer_projected_distance = np.dot(inner_centre_to_outer_nodes, distance/dis_mag)
    outer.dot_dis_nor.x.array[:] *= outer_projected_distance[:]
    # The nucleus pulls the part of the cell in front (in the direction of the distance vector) of the nucleus centre
    outer_mask = outer_projected_distance < 0.0
    outer.dot_dis_nor.x.array[outer_mask] = 0.0
    # Force proportional to projected distance
    inner_centre_to_inner_nodes = inner.nodes - inner_centre
    inner_projected_distance = np.dot(inner_centre_to_inner_nodes, distance/dis_mag)
    inner.dot_dis_nor.x.array[:] *= inner_projected_distance[:]
    # The plasma membrane pulls the part of the nucleus in front of the nucleus centre
    inner_mask = inner_projected_distance < 0.0
    inner.dot_dis_nor.x.array[inner_mask] = 0.0

    # Multiply the forces to ensure the total force
    inner_dot_dis_nor = np.abs(assemble_scalar(inner.dot_dis_nor_form))
    outer_dot_dis_nor = np.abs(assemble_scalar(outer.dot_dis_nor_form))
    if dis_mag > 1.0e-6:
        inner_multiplier = total_force/inner_dot_dis_nor
        outer_multiplier = total_force/outer_dot_dis_nor
    else:
        inner_multiplier = 0.0
        outer_multiplier = 0.0
    inner.memb_nuen_spring.x.array[:] = inner_multiplier*inner.dot_dis_nor.x.array[:]
    outer.memb_nuen_spring.x.array[:] = outer_multiplier*outer.dot_dis_nor.x.array[:]
    return
# }}}

# Pressure gradient {{{
class PressureGradient(object):
    # Properties {{{
    @property
    def gradient(self):
        return self._gradient
    @property
    def direction(self):
        return self._direction
    # }}}
    # __init__ {{{
    def __init__(self, gradient, direction):
        self._gradient = gradient
        self._direction = direction
        return
    # }}}
    # __call__ {{{
    def __call__(self, gspde):
        # Initialisation of pressure
        pressure = np.zeros_like(gspde.pressureGradient.x.array)
        # Get normal array
        normalArray = FromVectorToMatrix(gspde.normal.x.array, gspde.dimSpa)
        # Compute dot product between normal and direction
        normal_dot_direction = np.dot(normalArray, self.direction)
        # Only consider points whose normal is opposite to the direction
        mask = normal_dot_direction < 0.0
        # Pressure is then gradient*normal\cdot direction
        pressure[mask] = self.gradient*normal_dot_direction[mask]
        return pressure
    # }}}
# }}}
