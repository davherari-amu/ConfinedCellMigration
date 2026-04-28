# Libraries {{{
from mpi4py import MPI

from .core import GSPDE
from .forces import (ContactInnerOuterSurfaces, ContactOuterOuterSurfaces,
                     ContactWithClosedSurface,
                     SpringForceCentroidToCentroid)
from .Dynamics import MembraneNucleusDynamics

import numpy as np
from pdb import set_trace
from shapely.geometry import Polygon

from ..misc_utils import mprint
# }}}

# Cell class {{{
class Cell(object):
    # Description {{{
    """
    Class to consider cells as plasma-membrane plus nuclear envelope
    """
    # }}}
    # Properties {{{
    @property
    def memb(self):
        return self._memb
    @property
    def nuen(self):
        return self._nuen
    @property
    def contact_stiffness(self):
        return self._contact_stiffness
    @property
    def contact_tol(self):
        return self._contact_tol
    # }}}
    # __init__ {{{
    def __init__(self, memb_params, nuen_params, contact_stiffness,
                 contact_tol = 0.0):
        self._memb = GSPDE(memb_params)
        self._nuen = GSPDE(nuen_params)
        self._memb.centroid_function = lambda: self._nuen.centroid
        self._contact_stiffness = contact_stiffness
        self._contact_tol = contact_tol
        # Check subclass of MembraneNucleusDynamics
        if not isinstance(self.memb.Dynamics, MembraneNucleusDynamics):
            message = f"Invalid dynamics: {type(self.memb.Dynamics).__name__} must inherit from MembraneNucleusDynamics"
            raise TypeError(message)
        if not isinstance(self.nuen.Dynamics, MembraneNucleusDynamics):
            message = f"Invalid dynamics: {type(self.nuen.Dynamics).__name__} must inherit from MembraneNucleusDynamics"
            raise TypeError(message)
        return
    # }}}
    # Write results {{{
    def WriteResults(self, t):
        self.memb.WriteResults(t)
        self.nuen.WriteResults(t)
        return
    # }}}
    # Close results {{{
    def CloseResults(self):
        self.memb.CloseResults()
        self.nuen.CloseResults()
        return
    # }}}
    # Solve iteration {{{
    def SolveIteration(self, t):
        self.ForceInteraction()
        self.nuen.SolveIteration(t)
        self.memb.SolveIteration(t)
        return
    # }}}
    # Force interaction {{{
    def ForceInteraction(self):
        tensionRatio = self.nuen.tensionRatio.value
        # Compute force
        contNuen, contMemb = ContactInnerOuterSurfaces(self.nuen, self.memb,
                                                       self.contact_stiffness,
                                                       tol = self.contact_tol)
        # Scale nucleus contact
        contNuen = contNuen/tensionRatio
        # Spring force
        SpringForceCentroidToCentroid(self.nuen, self.memb)
        # Assign values
        self.nuen.memb_nuen_force.x.array[:] = contNuen[:]
        self.memb.memb_nuen_force.x.array[:] = contMemb[:]
        return
    # }}}
    # Write report {{{
    def WriteReport(self, t):
        self.memb.WriteReport(t)
        self.nuen.WriteReport(t)
        return
    # }}}
# }}}
