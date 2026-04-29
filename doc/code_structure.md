# Code structure

This document describes the organisation of the source code in `src/`. It is intended for readers who wish to understand, extend, or modify the framework.

---

## Entry point

### `src/single_cell.py`

The main script. It reads the JSON parameter file, builds all objects, and runs the time-stepping loop. Its responsibilities are:

1. Parse and organise parameters from the JSON file.
2. Build the cell mesh(es) using `gmsh`.
3. Instantiate the dynamics classes by combining them with `CombineClasses` (see below).
4. Create a `GSPDE` object (PM only) or a `Cell` object (PM + NE).
5. Advance the solution in time, calling `SolveIteration` at each step and writing output.

No mathematical logic lives here; it is purely a configuration and orchestration layer.

---

## `src/python_utils/gspde/`

The core finite element library. It is structured around a small class hierarchy.

### `core.py` — `GSPDE`

The central class. Each `GSPDE` instance represents a single evolving curve (either the PM or the NE). It holds:
- The DOLFINx mesh and finite element spaces.
- A `Dynamics` object (see below) that owns all force definitions and the weak form.
- Utility methods: `GetPolygon`, `GetOrderedNodes`, `Equidistribute`, `SolveIteration`, `WriteResults`.

Geometric quantities (area, perimeter, centroid) are computed on-the-fly from the current mesh using `shapely`.

### `set_gspde.py`

Stateless setup functions called by `GSPDE.__init__`:
- `set_domain`: loads the gmsh model, extracts topology and node ordering.
- `set_measures`: defines UFL spatial coordinates, cell normals, and the integration measure.
- `set_fe_spaces`: creates the scalar, vector, and mixed finite element spaces.
- `set_nonlinear_problem`: assembles the DOLFINx `NonlinearProblem` and `NewtonSolver`.

### `Dynamics.py`

Defines the dynamics classes. Each class is responsible for declaring finite element functions, assembling force contributions into `totalForce_raw`, and updating state variables after each solve. The classes are designed to be **combined via multiple inheritance** using `CombineClasses` in `misc_utils.py`, so that any combination of physical effects can be activated from the parameter file without modifying the source code.

| Class | Responsibility |
|---|---|
| `BasicDynamics` | Base class. Defines core variables (position, curvature, normal), sets the weak form, manages output and self-repulsive contact. |
| `SizeControlDynamics` | Adds osmotic pressure and perimeter-dependent tension. |
| `RandomFilopodiaDynamics` | Adds active protrusion forces. |
| `MembraneNucleusDynamics` | Adds PM–NE contact and elastic spring forces as variables. |
| `NucleusToPMDynamics` | Extends `MembraneNucleusDynamics` with NE-specific viscosity, tension, and bending ratios in the weak form. |
| `ECMBarrierPointDynamics` | Adds contact with circular (point) obstacles. |
| `ECMBarrierPolygonDynamics` | Adds contact with polygon obstacles. |
| `ExternalPressureGradient` | Adds a passive external pressure gradient. |

The weak form is set in `BasicDynamics.set_weak_form`, with the velocity equation modified by each subclass through `totalForce_raw` and `barrierForce_raw`.

### `forces.py`

Contains all force computation functions and classes, operating directly on numpy arrays:

- `SelfRepulsiveForce_kdtree`: self-contact force using a KD-tree.
- `eaSumNodal` / `ea_Dotpro`: active protrusion force definition and space–time distribution.
- `ContactInnerOuterSurfaces`: PM–NE contact (inner/outer topology).
- `ContactToECMPoints` / `ContactToECMPolygons`: contact with external obstacles.
- `SpringForceCentroidToCentroid`: PM–NE elastic coupling force.
- `PressureGradient`: passive pressure gradient.

### `multi_gspde.py` — `Cell`

A thin wrapper combining a PM `GSPDE` and an NE `GSPDE`. At each time step it:
1. Computes the PM–NE contact and spring forces (`ForceInteraction`).
2. Advances the NE, then the PM (`SolveIteration`).

The sequential solve (NE before PM) reflects the fact that the NE position informs the elastic force on the PM.

### `curvature.py`

Provides `InitialCurvature`, which computes the initial mean curvature field by solving a separate variational problem at $t=0$.

### `ecm_funcs.py`

Helper functions for building obstacle geometries from parameter file inputs:
- `ECM_from_list`: point obstacles from a coordinate list.
- `ECMPolygons`: polygon obstacles from a list of vertex lists.

---

## `src/python_utils/`

Utility modules used by `single_cell.py` and `Dynamics.py`.

### `mesh_utils.py`

- `MakeCircle`, `MakeSandClock`: gmsh geometry builders.
- `EquidistributeMesh`: mesh tangential redistribution after each time step, preventing element distortion during large deformations.
- `ArcLengthSpline`: arc-length parameterisation used by the equidistribution algorithm.
- `OrderNodeList`: returns a globally ordered list of node indices, needed to traverse the curve consistently.

### `misc_utils.py`

- `CombineClasses`: creates a new class inheriting from multiple named classes, enabling the dynamics composition pattern.
- `SelectFunction` / `SelectClass`: look up a function or class by name from a module, used to translate string entries in the JSON file to Python objects.
- `mprint`: MPI-safe print (rank 0 only).
- `PlotCircles`, `PlotPolygons`: write obstacle geometry to VTK for visualisation.

### `output_utils.py` — `Output`

Manages the DOLFINx `VTKFile` writer. Each `GSPDE` holds one `Output` instance.

---

## Adding a new force

To add a new force contribution:

1. **Implement the force** as a function or class in `forces.py`, operating on numpy arrays or DOLFINx `Function` objects.
2. **Create a new `Dynamics` subclass** in `Dynamics.py`. In `add_variables`, declare any new DOLFINx `Function` objects and append them to `output_variables`. In `add_expressions`, add the force to `totalForce_raw` or `barrierForce_raw`. In `UpdateLoads`, call your force computation.
3. **Register the new dynamics name** in the `cell_dynamics` or `nucl_dynamics` lists in `single_cell.py`, or add it as a `"dynamics"` entry in the relevant JSON block so that `CombineClasses` picks it up.
