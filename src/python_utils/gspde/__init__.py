"""
GSPDE package
=============
This package contains modular components for defining and solving
generalised surface PDEs with different protrusive and mechanical forces.

Submodules:
-----------
- core: The main GSPDE class (handles setup and solving).
- forces: Protrusion and repulsive force definitions.
- domain, measures, fe_spaces, variables, solver: Setup routines.

Example:
--------
from python_utils.gspde import GSPDE

gspde = GSPDE(model=my_model, quadrature_degree=2, meshOrder=2)
"""

from .core import GSPDE

__all__ = ["GSPDE"]
