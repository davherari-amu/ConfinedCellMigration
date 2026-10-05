# ConfinedCellMigration

A geometric surface PDE framework for simulating cell migration, coupling the plasma membrane (PM) and the nuclear envelope (NE) through biophysically motivated forces.

This code accompanies the manuscript:

> **Membrane tension and cell–nucleus mechanical coupling regulate confined cell migration**  
> David Hernandez-Aristizabal, Anotida Madzvamuse, Rachele Allena  
> *Under review*, 2026

---

## Overview

The framework models the PM and NE as evolving closed curves in 2D, governed by a system of geometric surface partial differential equations (GS-PDEs) derived from the Helfrich energy functional. The numerical solution uses the evolving surface finite element method, implemented with [FEniCSx](https://fenicsproject.org/) (DOLFINx 0.9.0).

The model accounts for:
- Curvature-dependent bending and tension forces (Helfrich energy)
- Osmotic pressure for area regulation
- Perimeter-dependent membrane tension for surface area regulation
- Active forces representing actin-driven protrusions
- Elastic PM–NE coupling (LINC complex)
- Passive external pressure
- Repulsive contact forces (self-contact, PM–NE contact, contact with external obstacles)

## Repository structure

```
ConfinedCellMigration/
├── doc/                        # Documentation
│   ├── installation.md
│   ├── usage.md
│   ├── parameters.md
│   └── code_structure.md
├── examples/                   # Parameter files for each simulation in the paper
│   ├── params_mecha_cons.json      # Mechanical consistency (Fig 4)
│   ├── params_peri_cont.json       # Perimeter control (Fig 5)
│   ├── params_migr_with_nucl.json  # Migration with nucleus (Fig 6)
│   ├── params_conf_disc.json       # Active migration through disc obstacles (Figs 7 and 8)
│   ├── params_conf_channel.json    # Passive migration through a channel (Fig 9)
│   ├── params_transient_release.json      # Channel crossing, transient release of S (Fig 10)
│   └── params_delayed_reinforcement.json  # Channel crossing, delayed reinforcement of S (Fig 10)
├── src/
│   ├── single_cell.py          # Main entry point
│   └── python_utils/
│       ├── gspde/              # Core GS-PDE library
│       ├── mesh_utils.py
│       ├── misc_utils.py
│       └── output_utils.py
├── package-fenics-env.txt      # Exact conda environment snapshot
├── CITATION.cff                # Citation metadata (read by GitHub and Zenodo)
└── LICENSE
```

## Quick start

```bash
# 1. Create the conda environment
conda create --name fenics-env --file package-fenics-env.txt

# 2. Activate it
conda activate fenics-env

# 3. Run an example from the examples/ directory
cd examples/
python3 ../src/single_cell.py -i params_mecha_cons.json
```

Results are written to `examples/results/` as VTK files, readable with [ParaView](https://www.paraview.org/).

See [`doc/installation.md`](doc/installation.md) and [`doc/usage.md`](doc/usage.md) for full instructions.

## Requirements

| Package | Version |
|---|---|
| Python | 3.13 |
| FEniCSx (DOLFINx) | 0.9.0 |
| PETSc | 3.22.2 |
| petsc4py | 3.22.1 |
| mpi4py | 4.0.1 |
| numpy | 2.2.0 |
| scipy | 1.14.1 |
| shapely | 2.1.0 |
| gmsh | 4.13.1 |
| matplotlib | 3.10.0 |
| MPICH | 4.2.3 |

A full exact environment snapshot is provided in `package-fenics-env.txt`.

## Licence

This project is licensed under the MIT Licence — see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code, please cite the accompanying manuscript (reference to be updated upon publication):

```
Hernandez-Aristizabal, D., Madzvamuse, A., Allena, R. (2026).
Membrane tension and cell–nucleus mechanical coupling regulate confined cell migration. Under review.
```

## Contact

David Hernandez-Aristizabal  
Université Côte d'Azur, LJAD UMR CNRS 7351, Nice, France  
david.hernandez-aristizabal@univ-cotedazur.fr  
ORCID: [0000-0003-4041-3785](https://orcid.org/0000-0003-4041-3785)

## Acknowledgment
This project has been supported by the Institut national de la santé et de la recherche médicale (INSERM) through project INSERM-MIC2024-LSP 296457, led by Rachele Allena and Frédéric Luton.
