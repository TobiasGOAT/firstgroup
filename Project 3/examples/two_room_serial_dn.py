# examples/two_room_serial_dn.py
"""
Serial Dirichlet–Neumann (DN) scheme for three subdomains:

    Ω1  --Γ1--  Ω2  --Γ2--  Ω3

Notes
-----
• For stable starts, keep a small flux relaxation (beta) inside the
  orchestrator (e.g. 0.1 or scaled by dx). We keep omega (Dirichlet
  relaxation on traces) configurable here via DNParams.
• Optional "tall" middle room: Ly(Ω2)=2.0. If your Interface class supports
  spanning (left_span/right_span), Γ1 maps the lower half of Ω2's left edge,
  and Γ2 maps the upper half of Ω2's right edge.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Allow running this file directly (module imports from project root)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ddm.subdomain import RectSubdomain
from src.ddm.interface import Interface
from src.ddm.orchestrator_dn import OrchestratorDN, DNParams

# ----------------------------- Configuration -----------------------------
DX = 1.0 / 20.0     # grid step; Ly=1.0 -> 21 nodes; Ly=2.0 -> 41 nodes
OMEGA = 0.6         # relaxation for Dirichlet trace updates (DNParams.omega)
USE_TALL_O2 = True  # use Ω2 with Ly=2.0 and interface spans
SHOW_FIGS = True    # draw figures at the end

# ----------------------------- Helpers -----------------------------------
def make_outer_dirichlet(sd: RectSubdomain, sides_to_val: dict):
    """
    Build constant Dirichlet arrays per side.

    Parameters
    ----------
    sd : RectSubdomain
    sides_to_val : dict
        e.g. {"bottom": 15, "left": 40, "top": 15}. Missing or None -> no BC.

    Returns
    -------
    dict[side] -> np.ndarray | None
    """
    out = {}
    for side in ("bottom", "left", "top", "right"):
        val = sides_to_val.get(side, None)
        if val is None:
            out[side] = None
        else:
            n = sd.side_length_nodes(side)
            out[side] = np.full(n, float(val))
    return out


def make_outer_neumann(sd: RectSubdomain, sides_to_flux: dict):
    """
    Build constant Neumann arrays per side (kept for completeness).

    Parameters
    ----------
    sd : RectSubdomain
    sides_to_flux : dict
        e.g. {"left": 0.0}. Missing or None -> no BC.

    Returns
    -------
    dict[side] -> np.ndarray | None
    """
    out = {}
    for side in ("bottom", "left", "top", "right"):
        val = sides_to_flux.get(side, None)
        if val is None:
            out[side] = None
        else:
            n = sd.side_length_nodes(side)
            out[side] = np.full(n, float(val))
    return out


def plot_subdomain(sd: RectSubdomain, ax, title: str):
    """
    Filled-contour plot of the last solution stored in the subdomain.
    """
    u = sd.last_u.reshape((sd.Ny, sd.Nx))
    x = np.linspace(0, sd.Lx, sd.Nx)
    y = np.linspace(0, sd.Ly, sd.Ny)
    X, Y = np.meshgrid(x, y)
    h = ax.contourf(X, Y, u, levels=40)
    ax.set_title(title, fontsize=14)
    return h


# ----------------------------- Main DN run -------------------------------
def main():
    # Geometry
    L2y = 2.0 if USE_TALL_O2 else 1.0

    O1 = RectSubdomain(Lx=1.0, Ly=1.0, dx=DX, name="Omega1",
                       external_dirichlet={}, external_neumann={})
    O2 = RectSubdomain(Lx=1.0, Ly=L2y, dx=DX, name="Omega2",
                       external_dirichlet={}, external_neumann={})
    O3 = RectSubdomain(Lx=1.0, Ly=1.0, dx=DX, name="Omega3",
                       external_dirichlet={}, external_neumann={})

    # Outer BCs (classic “two-room” setup)
    # Top/bottom = 15°C; Ω1 left (radiator) = 40°C; Ω3 right (window) = 5°C.
    O1.external_dirichlet = make_outer_dirichlet(O1, {"bottom": 15, "left": 40, "top": 15})
    O2.external_dirichlet = make_outer_dirichlet(O2, {"bottom": 15, "top": 15})
    O3.external_dirichlet = make_outer_dirichlet(O3, {"bottom": 15, "top": 15, "right": 5})

    O1.external_neumann = make_outer_neumann(O1, {})
    O2.external_neumann = make_outer_neumann(O2, {})
    O3.external_neumann = make_outer_neumann(O3, {})

    # Interfaces
    if USE_TALL_O2:
        # Γ1: Ω1 right (21 nodes) ↔ lower half of Ω2 left (21 of 41 nodes)
        # Γ2: upper half of Ω2 right (21 nodes) ↔ Ω3 left (21 nodes)
        G1 = Interface(
            left=O1, right=O2, left_side="right", right_side="left",
            left_span=(0, O1.side_length_nodes("right")),
            right_span=(0, O1.side_length_nodes("right")),
        )
        G2 = Interface(
            left=O2, right=O3, left_side="right", right_side="left",
            left_span=(O2.side_length_nodes("right") - O3.side_length_nodes("left"),
                       O2.side_length_nodes("right")),
            right_span=(0, O3.side_length_nodes("left")),
        )
    else:
        # Square rooms: spans are not needed (full-edge coupling).
        G1 = Interface(left=O1, right=O2, left_side="right", right_side="left")
        G2 = Interface(left=O2, right=O3, left_side="right", right_side="left")

    # DN orchestration
    params = DNParams(omega=OMEGA, max_iter=200, tol=1e-6)
    orch = OrchestratorDN(omega2=O2, omega1=O1, omega3=O3,
                          gamma1=G1, gamma2=G2, params=params)

    result = orch.iterate()
    print(f"Converged in {len(result['err_history'])} iterations; "
          f"final err = {result['err_history'][-1]:.3e}")

    # Final solves for plotting (apply converged Dirichlet traces)
    O2.clear_interface_bcs()
    G1.set_dirichlet_on_right(result["trace_G1"])
    G2.set_dirichlet_on_left(result["trace_G2"])
    O2.solve("dirichlet")

    O1.clear_interface_bcs()
    G1.set_dirichlet_on_left(result["trace_G1"])
    O1.solve("dirichlet")

    O3.clear_interface_bcs()
    G2.set_dirichlet_on_right(result["trace_G2"])
    O3.solve("dirichlet")

    # (Optionally) one more Ω2 solve to refresh caches after Ω1/Ω3 updates
    O2.solve("dirichlet")

    # Plots
    if SHOW_FIGS:
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
        h1 = plot_subdomain(O1, axs[0], "Ω1")
        h2 = plot_subdomain(O2, axs[1], "Ω2")
        h3 = plot_subdomain(O3, axs[2], "Ω3")
        fig.colorbar(h2, ax=axs.ravel().tolist(), label="Temperature")
        plt.show()

        plt.figure(figsize=(6, 4.2))
        plt.semilogy(result["err_history"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel(r"$\|\Delta u_{\Gamma}\|_\infty$")
        plt.title("DN Convergence")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.show()


if __name__ == "__main__":
    main()
