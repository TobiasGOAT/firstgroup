# Project 3 — Dirichlet–Neumann Room Heating

A small, structured codebase for solving steady 2D heat conduction with a single-domain solver and a serial Dirichlet–Neumann (DN) domain decomposition across three rectangular subdomains (Ω₁–Ω₂–Ω₃).  
It also contains a reproducible script for **Task 1** to print/plot and export the small matrices at mesh size Δx = 1/3.

---

## 1) Repository layout

```
.
├─ examples/
│  ├─ task1_matrix_tables.py        # Task 1: build Ω1, Ω2, Ω3 matrices (Δx=1/3), pretty print + CSV + plots
│  └─ two_room_serial_dn.py         # Serial DN runner for three rooms (Ω1–Ω2–Ω3), supports tall Ω2 (Ly=2.0)
│
├─ src/
│  ├─ heat/
│  │  └─ heatSolver.py              # Single-domain 2D Poisson/heat solver (build K,b; solve; return boundary traces)
│  └─ ddm/
│     ├─ subdomain.py               # RectSubdomain wrapper around heatSolver with external/interface BC management
│     ├─ interface.py               # (Expected) coupling of two RectSubdomain edges, optional spans for tall Ω2
│     └─ orchestrator_dn.py         # (Expected) DN orchestrator + DNParams (omega,tol,max_iter), iterate()
│
├─ reports/
│  └─ task1/                        # Auto-generated artifacts for Task 1 (CSV, PNG)
│     └─ .gitkeep
└─ requirements.txt                 # Python dependencies (NumPy, SciPy, pandas, matplotlib, …)
```

> **Note**: If `src/ddm/interface.py` and `src/ddm/orchestrator_dn.py` are not yet in the repo, add them following the API described below. The examples assume their presence.

---

## 2) Quick start

### Environment
```bash
# optional: create a venv
python -m venv .venv
# Windows: .venv\Scripts\activate   |  Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

### Task 1 (matrices, tables, CSV, plots)
```bash
python -m examples.task1_matrix_tables
```
Outputs go to:
```
reports/task1/
  Omega1_K.csv, Omega1_b.csv, Omega1_K_values.png
  Omega2_K.csv, Omega2_b.csv, Omega2_K_values.png
  Omega3_K.csv, Omega3_b.csv, Omega3_K_values.png
```

### DN example (three rooms, optional tall Ω₂)
```bash
python -m examples.two_room_serial_dn
```
Shows: 3 subdomain temperature contours + DN convergence curve (`||Δu_Γ||∞`).

---

## 3) What each file does (inputs & outputs)

### `examples/task1_matrix_tables.py`
**Purpose**: Implements **Task 1**. With Δx=1/3 ⇒ `Nx=Ny=4` in Ω₁ and Ω₃, and `Nx=4, Ny=7` in Ω₂.  
Builds K and b for Ω₁, Ω₂, Ω₃ **before/after BC application** via `heatSolver`, then:
- prints neat ASCII tables to console (fallback if terminal is non-UTF8),
- prints pandas DataFrames to console,
- saves K and b to CSV,
- saves matrix heatmaps (values) as PNG.

**Inputs**:
- Fixed mesh: `DX = 1/3`.
- BCs:
  - Ω₁: Dirichlet bottom=15, left=40, top=15; **Neumann right = 0**.
  - Ω₂: Dirichlet bottom=15, top=15; Dirichlet on both vertical interfaces (to be provided by Ω₁/Ω₃ in DN, but here fixed to zeros just to materialize rows).
  - Ω₃: Dirichlet bottom=15, top=15, right=5; **Neumann left = 0**.

**Outputs**:
- Console: formatted tables of K and b for each Ω.
- `reports/task1/*.csv`: labeled CSVs (`r0..`, `c0..`, `b0..`).
- `reports/task1/*_values.png`: heatmaps of K entries (for visual structure/weights).

---

### `examples/two_room_serial_dn.py`
**Purpose**: Serial DN coupling for three rooms. Also supports **tall Ω₂** (`Ly=2.0`) with interface **spans**:
- Γ₁ couples Ω₁ right edge ↔ lower half of Ω₂ left edge.
- Γ₂ couples upper half of Ω₂ right edge ↔ Ω₃ left edge.

**Inputs** (config at top of file):
- `DX` (grid step, default `1/20`),
- `OMEGA` (Dirichlet relaxation for trace updates),
- `USE_TALL_O2` (bool),
- fixed outer Dirichlet walls: top/bottom=15; Ω₁ left=40; Ω₃ right=5.

**Outputs**:
- Prints iterations and final error.
- Two figures: (1) temperatures on Ω₁,Ω₂,Ω₃; (2) semilogy convergence.

---

### `src/heat/heatSolver.py`
**Purpose**: Single-domain 2D heat/Poisson solver on a rectangular grid.

**Minimal API (used by examples)**:
```python
# Construction
hs = heatSolver(dx, [Lx, Ly], D, N)

# Attributes after assembly
hs.K   # sparse/dense system matrix (after BC application)
hs.b   # right-hand side vector

# Solve + boundary traces
u_flat, side_vals = hs.solve(return_dirichlet=True)
# side order: [bottom, left, top, right]
# if return_dirichlet=False -> returns Neumann flux traces instead
```

**Inputs**:
- `dx`: mesh size; domain spans `[0,Lx]×[0,Ly]`.
- `D`: Dirichlet lists `[bottom, left, top, right]`, each `None` or 1D array of length matching that side.
- `N`: Neumann lists `[bottom, left, top, right]`, `None` or 1D array (flux values).  
  (For zero-flux sides pass `np.zeros(n_side)`.)

**Outputs**:
- `(u_flat, side_vals)` from `solve`, with `u_flat` flattened by row/column ordering used in `heatSolver`.
- Assembled `K`, `b` accessible for reporting.

---

### `src/ddm/subdomain.py`
**Purpose**: Thin wrapper around `heatSolver` adding **external** and **interface** BC management per side.

**Key class**: `RectSubdomain(Lx, Ly, dx, name, ...)`

**Important methods**:
- `side_length_nodes(side) -> int`
- `set_interface_dirichlet(side, values)`
- `set_interface_neumann(side, flux)`
- `clear_interface_bcs()`
- `solve(boundary_mode_return="dirichlet") -> (u_flat, {side -> 1D array})`
- `get_boundary_values(mode="dirichlet"|"neumann")`

**State**:
- `last_u` (latest solution, flattened),
- cached `last_boundary_dirichlet` / `last_boundary_flux`.

---

### `src/ddm/interface.py`  *(expected)*
**Purpose**: Couples two `RectSubdomain` edges. Provides optional **spans** for unequal edge lengths (e.g., tall Ω₂).

**Typical API**:
```python
Interface(
    left: RectSubdomain, right: RectSubdomain,
    left_side: Literal["bottom","left","top","right"],
    right_side: Literal["bottom","left","top","right"],
    left_span:  Tuple[int,int] | None = None,   # [start, end) along the chosen side
    right_span: Tuple[int,int] | None = None,
)

# Apply Dirichlet traces across the interface:
iface.set_dirichlet_on_left(values_1d)
iface.set_dirichlet_on_right(values_1d)

# Get samples (for DN updates) if needed:
vals = iface.sample_left_dirichlet()
vals = iface.sample_right_dirichlet()
```

**Inputs/Outputs**: 1D arrays aligned to the selected span length.

---

### `src/ddm/orchestrator_dn.py`  *(expected)*
**Purpose**: Runs the **serial Dirichlet–Neumann iteration** across Γ₁ and Γ₂.

**Typical API**:
```python
@dataclass
class DNParams:
    omega: float = 0.6    # relaxation on Dirichlet trace
    tol: float = 1e-6
    max_iter: int = 200
    # (Inside implementation keep a small beta for flux stabilization, e.g. 0.1.)

class OrchestratorDN:
    def __init__(self, omega1, omega2, omega3, gamma1, gamma2, params: DNParams): ...
    def iterate(self) -> dict:
        # returns {
        #   "err_history": [e0, e1, ...],
        #   "trace_G1": np.ndarray,   # converged Dirichlet on Γ1 as seen by Ω1/Ω2
        #   "trace_G2": np.ndarray    # converged Dirichlet on Γ2 as seen by Ω2/Ω3
        # }
```

---

## 4) How this maps to the coursework tasks

- **Task 1 — Small matrices for Ω₁, Ω₂, Ω₃ with Δx=1/3**  
  `examples/task1_matrix_tables.py` produces console tables, CSVs, and plots.  
  Use the CSVs directly in the report (they are labeled `r*`/`c*` and easy to cite).

- **Task 2 — Heating adequacy / qualitative analysis (later)**  
  Use `two_room_serial_dn.py` to compute a realistic temperature field with the DN coupling and inspect whether temperatures near the “window” (Ω₃ right) fall below comfort thresholds. (Add post-processing as needed.)

- **Task 3 — Temperature distribution plot**  
  The DN example already produces filled-contour plots for each subdomain; stitch them or present separately as required.

---

## 5) Reproducibility & Tips

- **Console encoding**: If your terminal is not UTF-8, the Task 1 script automatically falls back to ASCII box drawing. Pandas tables and CSVs are unaffected.
- **Where are artifacts saved?**  
  `reports/task1/` (created automatically). Clean by removing the folder.
- **Matplotlib backend**: If running on a headless server, set `MPLBACKEND=Agg` and skip `plt.show()`—images are still saved.
- **Extending BCs**: Use `RectSubdomain.external_dirichlet/neumann` for outer walls and `set_interface_dirichlet/ neumann` for Γ-coupling.
- **Mesh changes**: For Task 1 keep `DX=1/3`. For DN demo, adjust `DX` for accuracy vs. runtime.

---

## 6) Contributing / commit style

- **Commit titles**: `feat(module): short action`, `fix`, `refactor`, `docs`, `chore`.
- **Examples**  
  - `feat(examples): add DN runner with tall Ω2 support`  
  - `docs(task1): export K,b to CSV and add heatmaps`
- **PRs**: Target the feature branch (e.g., `proj_3`) and open a PR to `main` when stable.

---

## 7) License / ownership

Coursework code for the “Advanced Course in Numerical Algorithms with Python/SciPy”. Reuse within the group allowed; cite if used elsewhere.

---

### Appendix — Running from repo root

```bash
# Task 1 matrices/tables/CSV/plots
python -m examples.task1_matrix_tables

# DN 3-room demo
python -m examples.two_room_serial_dn
```
