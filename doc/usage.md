# Usage

## Running a simulation

All simulations are launched via the main entry point `src/single_cell.py`, which takes a single JSON parameter file as input. The intended workflow is to run from the `examples/` directory:

```bash
conda activate fenics-env
cd examples/
python3 ../src/single_cell.py -i <params_file.json>
```

For example, to run the mechanical consistency test from paper:

```bash
cd examples/
python3 ../src/single_cell.py -i params_mecha_cons.json
```

---

## Output

Results are written to a `results/` subdirectory inside `examples/`, organised by the folder name specified in the parameter file under the `"files"` key. For instance, `params_mecha_cons.json` writes to:

```
examples/results/dimless_tests/mechanical_consistency/area_stiffness_32/
```

Each run produces:
- **VTK files** (`.pdv` format, readable by ParaView) containing the mesh geometry and all field variables at each saved time step. Two sets are written: one for the PM (`ce_`) and one for the NE (`nu_`) when the nucleus is included.
- **CSV files** containing scalar time-series data (area, perimeter, energies, force norms, centroid position, velocity) written at every reported time step.
- A copy of the parameter file (`params.json`) for traceability.

### Field variables in the VTK output

| Name | Description |
|---|---|
| `u` | Position (displacement from initial configuration) |
| `H` | Signed mean curvature |
| `normal` | Outward-pointing unit normal |
| `Fs` | Self-repulsive contact force |
| `Fb` | Barrier force (contact with NE or obstacles) |
| `Ftot` | Total external force |
| `opre` | Osmotic pressure + perimeter control force |
| `ea` | Active (protrusion) force magnitude |
| `memb_nuen_force` | PM–NE contact force |
| `memb_nuen_spring` | PM–NE elastic force |

### CSV time-series columns

| Column | Description |
|---|---|
| `time` | Simulation time (dimensionless) |
| `centroid_x`, `centroid_y` | Centroid coordinates |
| `area` | Enclosed area |
| `perimeter` | Curve perimeter |
| `energy` | Total Helfrich energy |
| `bending_energy` | Pure bending energy |
| `force_osmo` | L2 norm of osmotic pressure force |
| `force_ea` | L2 norm of active protrusion force |
| `force_memb_nuen` | L2 norm of PM–NE contact force |
| `force_memb_nuen_spring` | L2 norm of PM–NE elastic force |

---

## Examples in the paper

The table below maps each example in `examples/` to the corresponding figure in the paper.

| Parameter file | Description |
|---|---|
| `params_mecha_cons.json` | Mechanical consistency: curve-shortening flow with and without osmotic pressure |
| `params_peri_cont.json` | Effect of perimeter control on randomly migrating cell |
| `params_migr_with_nucl.json` | Directed migration with nucleus, varying PM–NE elasticity |
| `params_conf_disc.json` | Active migration through two-disc confinement |
| `params_conf_channel.json` | Passive migration through a confining channel |

---

## Modifying a simulation

The simplest way to explore different configurations is to copy an existing parameter file and modify the relevant entries. See [`parameters.md`](parameters.md) for a full description of all parameters.

For instance, to change the PM–NE elasticity in the directed migration example, copy `params_migr_with_nucl.json` and modify the `"spring_stiffness"` entry under `"nucleus"`.
