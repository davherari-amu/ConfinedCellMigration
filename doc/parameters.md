# Parameter reference

All simulations are controlled by a JSON parameter file passed to `single_cell.py` with the `-i` flag. The file is organised into the following top-level blocks.

For the mathematical definitions of each quantity, refer to the paper.

---

## `time`

Controls the temporal discretisation.

| Key | Type | Description |
|---|---|---|
| `Ttot` | float | Total simulation time (dimensionless, in units of $\tau = \eta_c \bar{r}_c^2 / \sigma_c$) |
| `dt` | float | Time step |
| `print_each` | int | Write VTK output every this many time steps |
| `print_each_report` | int | Write CSV report every this many time steps |

---

## `cell`

Configuration for the plasma membrane (PM).

### `cell.mesh`

| Key | Type | Description |
|---|---|---|
| `radius` | float | Initial cell radius (dimensionless; always `1.0`, as this is the reference length) |
| `lc` | float | Mesh element size |
| `meshOrder` | int | Polynomial order of the mesh elements (`2` for quadratic, as in the paper) |
| `function` | string | Mesh generator function: `"MakeCircle"` (default) or `"MakeSandClock"` (for mechanical consistency test) |

### `cell.material`

| Key | Type | Description |
|---|---|---|
| `bendingTensionRatio` | float | $\mathrm{B}_c = k_c / (\sigma_c \bar{r}_c^2)$, bending-to-tension ratio of the PM |
| `area_stiffness` | float | $\Lambda_\alpha$, osmotic pressure penalisation parameter |
| `peri_stiffness` | float | $\Lambda_\beta$, perimeter control penalisation parameter |
| `peri_max_factor` | float | Reference perimeter as a multiple of the initial perimeter: $\bar{P}_c = \texttt{peri\_max\_factor} \times P_{c_0}$ |

### `cell.protrusions` *(optional)*

Include this block to add active protrusion forces (actin filaments).

| Key | Type | Description |
|---|---|---|
| `dynamics` | string | Must be `"RandomFilopodiaDynamics"` |
| `force_class` | string | Must be `"ea_Dotpro"` |
| `prot_num` | int | Number of protrusions (use either `prot_num` or `prot_freq`) |
| `prot_freq` | float | Mean number of protrusions per unit time (alternative to `prot_num`) |
| `prot_tim_min`, `prot_tim_max` | float | Minimum and maximum protrusion lifespan $T_a$ |
| `prot_mag_min`, `prot_mag_max` | float | Minimum and maximum protrusion force magnitude $F_a$ |
| `prot_wid_min`, `prot_wid_max` | float | Minimum and maximum protrusion width $l_a$ (arc length) |
| `prot_theta_pi_min`, `prot_theta_pi_max` | float | Angular range for protrusion placement, in units of $\pi$ (e.g., `0.0` to `2.0` for fully random, `0.4` to `0.6` for directed upwards) |
| `ea_params.direction` | string | `"normal"` (force along the outward normal) or `"source"` (force along the protrusion direction vector) |
| `ea_params.typ` | string | `"pressure"` (force per unit area) or `"force"` (total force) |
| `ea_params.ea_intersection` | string | How overlapping protrusions combine: `"max"` or `"sum"` |
| `ea_params.startAt` | float | Time at which the first protrusion starts |

### `cell.ecm` *(optional)*

Include this block to add external rigid obstacles.

| Key | Type | Description |
|---|---|---|
| `dynamics` | string | `"ECMBarrierPointDynamics"` (circular obstacles) or `"ECMBarrierPolygonDynamics"` (polygon obstacles) |
| `ecm_params.ecm_tol` | float | Contact tolerance $\varepsilon$ for obstacle contact (for point obstacles, this acts as the obstacle radius) |
| `ecm_make_func` | string | `"ECM_from_list"` (list of point coordinates) or `"ECMPolygons"` (list of polygon vertex lists) |
| `ecm_make_func_kwargs` | object | Arguments passed to the mesh generation function (see examples) |

### `cell.pressure` *(optional)*

Include this block to add a passive external pressure gradient.

| Key | Type | Description |
|---|---|---|
| `dynamics` | string | Must be `"ExternalPressureGradient"` |
| `class` | string | Must be `"PressureGradient"` |
| `class_kwargs.gradient` | float | Pressure gradient magnitude $F_p$ |
| `class_kwargs.direction` | list | Unit direction vector $\mathbf{u}_p$ as `[x, y]` |

---

## `nucleus` *(optional)*

Include this block to add the nuclear envelope (NE). Omit it entirely for PM-only simulations.

| Key | Type | Description |
|---|---|---|
| `spring_stiffness` | float | $S$, dimensionless PM–NE elastic coupling stiffness |

### `nucleus.material`

| Key | Type | Description |
|---|---|---|
| `viscosityRatio` | float | $\mathrm{R}_\eta = \eta_c / \eta_n$, PM-to-NE viscosity ratio |
| `tensionRatio` | float | $\mathrm{R}_\sigma = \sigma_c / \sigma_n$, PM-to-NE tension ratio |
| `bendingRatio` | float | $\mathrm{R}_k = k_c / k_n$, PM-to-NE bending stiffness ratio |
| `peri_max_factor` | float | Reference NE perimeter as a multiple of the initial NE perimeter: $\bar{P}_n = \texttt{peri\_max\_factor} \times P_{n_0}$ |
| `radius` | float | Initial nucleus radius (dimensionless; default `0.5`) |

---

## `contact`

Controls the repulsive contact forces between all surfaces (PM self-contact, PM–NE contact, PM–obstacle contact).

| Key | Type | Description |
|---|---|---|
| `tol` | float | Contact tolerance $\varepsilon$ (contact activates when the signed distance falls below this value) |
| `stiffness` | float | $C$, contact stiffness (penalisation parameter) |

---

## `files`

| Key | Type | Description |
|---|---|---|
| `folder` | string | Relative path for output files, written under `results/` |

---

## `solver`

| Key | Type | Description |
|---|---|---|
| `quadrature_degree` | int | Quadrature degree for finite element integration (default `8`) |

---

## Annotated example

Below is a minimal annotated parameter file for a directed migration simulation with nucleus:

```json
{
    "time": {
        "Ttot": 2.0,          // Simulate for 2 relaxation times
        "dt": 1e-4,           // Time step
        "print_each": 200,    // Save VTK every 200 steps (every 0.02 tau)
        "print_each_report": 2
    },
    "cell": {
        "mesh": {
            "radius": 1.0,
            "lc": 0.02,        // ~314 elements on the circle perimeter
            "meshOrder": 2,
            "function": "MakeCircle"
        },
        "material": {
            "bendingTensionRatio": 1e-6,  // B_c, tension-dominated regime
            "area_stiffness": 32.0,        // Lambda_alpha
            "peri_stiffness": 10.0,        // Lambda_beta
            "peri_max_factor": 1.25        // P_bar_c = 1.25 * P_c0
        },
        "protrusions": {
            "dynamics": "RandomFilopodiaDynamics",
            "force_class": "ea_Dotpro",
            "prot_num": 1,
            "prot_tim_min": 2.0,           // Single protrusion lasting 2 tau
            "prot_mag_min": 8.0,           // F_a = 8 (dimensionless)
            "prot_wid_min": 4.0,           // l_a = 4 r_bar_c
            "prot_theta_pi_min": 0.5,      // Directed upwards (pi/2)
            "ea_params": {
                "direction": "normal",
                "startAt": 0.0,
                "typ": "pressure",
                "ea_intersection": "max"
            }
        }
    },
    "nucleus": {
        "spring_stiffness": 10.0,          // S = 10
        "material": {
            "viscosityRatio": 1.0,         // R_eta
            "tensionRatio": 0.1,           // R_sigma (NE 10x stiffer)
            "bendingRatio": 0.1            // R_k
        }
    },
    "contact": {
        "tol": 0.025,
        "stiffness": 10000.0
    },
    "files": {
        "folder": "my_simulation/"
    },
    "solver": {
        "quadrature_degree": 8
    }
}
```
