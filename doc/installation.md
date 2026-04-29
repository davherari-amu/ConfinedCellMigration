# Installation

## Requirements

This code relies on [FEniCSx](https://fenicsproject.org/) (DOLFINx 0.9.0), which is most reliably installed via [conda-forge](https://conda-forge.org/). Installation via pip or from source is not recommended, as FEniCSx depends on several compiled C++ libraries (PETSc, MPICH, MUMPS) that conda manages automatically.

The code has been tested on **Linux (x86-64)** with the exact environment specified in `package-fenics-env.txt`.

---

## Exact environment (recommended for full reproducibility)

This reproduces the exact package versions used to produce the results in the paper.

```bash
conda create --name fenics-env --file package-fenics-env.txt
conda activate fenics-env
```

This may take several minutes as it downloads a large number of packages.

---

## ParaView (for visualisation)

Results are written as VTK files. [ParaView](https://www.paraview.org/) (version 5.x or later) is recommended for visualisation. It is not included in the conda environment and should be installed separately from the ParaView website.
