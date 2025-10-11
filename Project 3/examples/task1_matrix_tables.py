import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

# --- I/O helpers -------------------------------------------------------------
def _to_dense(A):
    """Return a dense numpy array regardless of sparse/dense input."""
    return A.toarray() if sp.issparse(A) else np.asarray(A)

def _dense_b(b):
    """Return a dense 1D RHS vector from possibly-sparse input."""
    return np.asarray(b.todense()).ravel() if sp.issparse(b) else np.asarray(b).ravel()

def save_K_b_csv(name: str, K, b, outdir: str = "reports/task1"):
    """
    Persist K (matrix) and b (rhs) as clean CSV files with labels.
      - K -> <outdir>/<name>_K.csv (row labels r0.., column labels c0..)
      - b -> <outdir>/<name>_b.csv (single labeled column)
    """
    os.makedirs(outdir, exist_ok=True)
    Kd = _to_dense(K)
    bd = _dense_b(b)

    cols = [f"c{j}" for j in range(Kd.shape[1])]
    rows = [f"r{i}" for i in range(Kd.shape[0])]
    pd.DataFrame(Kd, index=rows, columns=cols).to_csv(
        os.path.join(outdir, f"{name}_K.csv"), encoding="utf-8", float_format="%.6g", index=True
    )

    pd.Series(bd, index=[f"b{i}" for i in range(len(bd))], name=f"{name}_b").to_csv(
        os.path.join(outdir, f"{name}_b.csv"), encoding="utf-8", float_format="%.6g", index=True
    )

# --- repo path so `src` is importable ----------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.heat.heatSolver import heatSolver  # solver

# --- configuration -----------------------------------------------------------
DX = 1.0 / 3.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "task1")
os.makedirs(OUT_DIR, exist_ok=True)

USE_ASCII_TABLE = True      # console boxed tables
USE_PANDAS_DF   = True      # DataFrame view + CSV export
USE_PLOTS       = True      # matrix heatmaps

SAVE_CSV = True             # CSV artifacts when DataFrame printing is on
SAVE_PNG = True             # PNG artifacts for plots

# --- console boxed-table rendering ------------------------------------------
def _supports_unicode_box():
    enc = (sys.stdout.encoding or "").lower()
    env = (os.environ.get("PYTHONIOENCODING", "")).lower()
    return ("utf" in enc) or env.startswith("utf")

def _render_table(title: str, A: np.ndarray, row_header: str) -> str:
    use_unicode = _supports_unicode_box()
    tl, tr, bl, br = ("┌", "┐", "└", "┘") if use_unicode else ("+", "+", "+", "+")
    h, v = ("─", "│") if use_unicode else ("-", "|")
    sep_l = "├" if use_unicode else "+"
    sep_r = "┤" if use_unicode else "+"
    sep_t = "┬" if use_unicode else "+"
    sep_b = "┴" if use_unicode else "+"

    rows, cols = A.shape
    col_w = [max(len(str(j)), 4) for j in range(cols)]
    head_w = max(len(row_header), 3)

    top = tl + h * (head_w + 2) + sep_t + sep_t.join(h * (w + 2) for w in col_w) + tr
    hdr = f" {row_header} ".rjust(head_w + 2) + v + "".join(
        f" {str(j).rjust(w)} {v}" for j, w in zip(range(cols), col_w)
    )
    mid = sep_l + h * (head_w + 2) + sep_t + sep_t.join(h * (w + 2) for w in col_w) + sep_r
    body = []
    for i in range(rows):
        row = [str(i).rjust(head_w)]
        for j, w in zip(range(cols), col_w):
            row.append(str(A[i, j]).rjust(w))
        body.append(" " + row[0] + " " + v + "".join(f" {c} {v}" for c in row[1:]))
    bottom = bl + h * (head_w + 2) + sep_b + sep_b.join(h * (w + 2) for w in col_w) + br

    return "\n".join([f"\n{title}", top, hdr, mid, *body, bottom])

def print_matrix_ascii(title: str, K, b):
    Kd = _to_dense(K)
    bd = _dense_b(b).reshape(1, -1)
    print(_render_table(f"{title}: matrix K  (shape={Kd.shape})", Kd, "i\\j"))
    print(_render_table(f"{title}: right-hand side b (as row)", bd, "b"))

# --- DataFrame view + CSV ----------------------------------------------------
def as_dataframe(K, b, nameK: str, nameb: str, save_csv: bool):
    Kd = _to_dense(K)
    bd = _dense_b(b)
    dfK = pd.DataFrame(Kd, index=[f"r{i}" for i in range(Kd.shape[0])],
                            columns=[f"c{j}" for j in range(Kd.shape[1])])
    sB = pd.Series(bd, index=[f"b{j}" for j in range(bd.shape[0])], name=nameb)

    with pd.option_context("display.width", 200, "display.max_columns", 200):
        print(f"\n{nameK} (shape={Kd.shape})")
        print(dfK)
        print(f"\n{nameb} (len={bd.shape[0]})")
        print(sB)

    if save_csv:
        dfK.to_csv(os.path.join(OUT_DIR, f"{nameK}.csv"), index=True, encoding="utf-8")
        sB.to_csv(os.path.join(OUT_DIR, f"{nameb}.csv"), index=True, encoding="utf-8")

# --- plotting ---------------------------------------------------------------
def plot_matrix(K, title: str, fname: str | None, use_structure: bool = False):
    import matplotlib.pyplot as plt
    Kd = _to_dense(K)

    plt.figure()
    if use_structure:
        plt.spy(Kd != 0, markersize=5)
        plt.title(f"{title} (structure)")
    else:
        plt.imshow(Kd, interpolation="none")  # neutral colormap per project rules
        plt.colorbar()
        plt.title(f"{title} (values)")
    plt.xlabel("column")
    plt.ylabel("row")
    plt.tight_layout()
    if fname and SAVE_PNG:
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.show()

# --- BC builders -------------------------------------------------------------
def build_bc_array(n: int, value: float | None):
    return None if value is None else np.full(n, float(value))

def build_omega1(dx: float):
    Nx = int(1 / dx) + 1
    Ny = Nx
    D = [build_bc_array(Nx, 15.0), build_bc_array(Ny, 40.0), build_bc_array(Nx, 15.0), None]
    N = [None, None, None, np.zeros(Ny)]      # Neumann on right (zero flux)
    return heatSolver(dx, [1.0, 1.0], D, N)

def build_omega3(dx: float):
    Nx = int(1 / dx) + 1
    Ny = Nx
    D = [build_bc_array(Nx, 15.0), None, build_bc_array(Nx, 15.0), build_bc_array(Ny, 5.0)]
    N = [None, np.zeros(Ny), None, None]      # Neumann on left (zero flux)
    return heatSolver(dx, [1.0, 1.0], D, N)

def build_omega2(dx: float):
    Nx = int(1 / dx) + 1         # 4
    Ny = int(2 / dx) + 1         # 7
    # Dirichlet on both vertical interfaces; outer bottom/top = 15
    D = [build_bc_array(Nx, 15.0), np.zeros(Ny), build_bc_array(Nx, 15.0), np.zeros(Ny)]
    N = [None, None, None, None]
    return heatSolver(dx, [1.0, 2.0], D, N)

# --- main workflow -----------------------------------------------------------
def main():
    hs1 = build_omega1(DX)
    hs3 = build_omega3(DX)
    hs2 = build_omega2(DX)

    # Console tables
    if USE_ASCII_TABLE:
        print_matrix_ascii("Omega1 (Neumann on right)", hs1.K, hs1.b)
        print_matrix_ascii("Omega3 (Neumann on left)",  hs3.K, hs3.b)
        print_matrix_ascii("Omega2 (Dirichlet on both interfaces)", hs2.K, hs2.b)

    # DataFrames (+ CSV)
    if USE_PANDAS_DF:
        as_dataframe(hs1.K, hs1.b, "Omega1_K", "Omega1_b", SAVE_CSV)
        as_dataframe(hs3.K, hs3.b, "Omega3_K", "Omega3_b", SAVE_CSV)
        as_dataframe(hs2.K, hs2.b, "Omega2_K", "Omega2_b", SAVE_CSV)

    # Plots
    if USE_PLOTS:
        plot_matrix(hs1.K, "Omega1 K", "Omega1_K_values.png", use_structure=False)
        plot_matrix(hs3.K, "Omega3 K", "Omega3_K_values.png", use_structure=False)
        plot_matrix(hs2.K, "Omega2 K", "Omega2_K_values.png", use_structure=False)
        # Optional structure plots:
        # plot_matrix(hs1.K, "Omega1 K", "Omega1_K_structure.png", use_structure=True)
        # plot_matrix(hs3.K, "Omega3 K", "Omega3_K_structure.png", use_structure=True)
        # plot_matrix(hs2.K, "Omega2 K", "Omega2_K_structure.png", use_structure=True)

    # CSV artifacts for the “named” version as well (mirrors DataFrame names)
    if SAVE_CSV:
        save_K_b_csv("Omega1", hs1.K, hs1.b, OUT_DIR)
        save_K_b_csv("Omega3", hs3.K, hs3.b, OUT_DIR)
        save_K_b_csv("Omega2", hs2.K, hs2.b, OUT_DIR)

if __name__ == "__main__":
    main()
