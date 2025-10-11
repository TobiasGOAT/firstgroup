# src/ddm/subdomain.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np

# Single–subdomain Laplace solver (already in your repo)
from src.heat.heatSolver import heatSolver

Side = Literal["bottom", "left", "top", "right"]
SIDES: tuple[Side, Side, Side, Side] = ("bottom", "left", "top", "right")


@dataclass
class RectSubdomain:
    """
    Thin wrapper around the single-domain finite-difference solver for a
    rectangular subdomain. It centralizes:
      - bookkeeping of external (outer) boundary conditions,
      - interface boundary data (overwritten each DN iteration),
      - cached solution and sampled boundary values.

    Notes
    -----
    • Dirichlet takes precedence over Neumann for any side (same rule as the
      underlying `heatSolver`).
    • All arrays provided to setters must match the number of nodes along the
      target side:
         bottom/top -> Nx nodes, left/right -> Ny nodes.
    • Outward normal convention matches `heatSolver`:
         bottom/left normals are negative, top/right are positive.
    """

    # geometry and discretization
    Lx: float
    Ly: float
    dx: float
    name: str

    # external (fixed) boundary conditions
    external_dirichlet: Dict[Side, Optional[np.ndarray]] = field(default_factory=dict)
    external_neumann: Dict[Side, Optional[np.ndarray]] = field(default_factory=dict)

    # interface boundary conditions (set each DN iteration by the orchestrator)
    iface_dirichlet: Dict[Side, Optional[np.ndarray]] = field(default_factory=dict)
    iface_neumann: Dict[Side, Optional[np.ndarray]] = field(default_factory=dict)

    # caches
    last_u: Optional[np.ndarray] = None  # flattened interior + boundary solution
    last_boundary_dirichlet: Dict[Side, Optional[np.ndarray]] = field(default_factory=dict)
    last_boundary_flux: Dict[Side, Optional[np.ndarray]] = field(default_factory=dict)

    # derived sizes
    Nx: int = field(init=False)
    Ny: int = field(init=False)

    # --------------------------------------------------------------------- #
    # lifecycle
    # --------------------------------------------------------------------- #
    def __post_init__(self) -> None:
        """Finalize sizes and ensure all BC dictionaries expose all four sides."""
        self.Nx = int(self.Lx / self.dx) + 1
        self.Ny = int(self.Ly / self.dx) + 1

        for d in (
            self.external_dirichlet,
            self.external_neumann,
            self.iface_dirichlet,
            self.iface_neumann,
            self.last_boundary_dirichlet,
            self.last_boundary_flux,
        ):
            for s in SIDES:
                d.setdefault(s, None)

    # --------------------------------------------------------------------- #
    # utilities
    # --------------------------------------------------------------------- #
    def side_length_nodes(self, side: Side) -> int:
        """Return the number of grid nodes that lie on a given boundary side."""
        if side in ("bottom", "top"):
            return self.Nx
        if side in ("left", "right"):
            return self.Ny
        raise ValueError(f"Unknown side: {side}")

    def _compose_dirichlet_list(self) -> List[Optional[List[float]]]:
        """
        Build the Dirichlet list in the order expected by `heatSolver`:
        [bottom, left, top, right].

        Priority: interface Dirichlet (if provided) overrides external Dirichlet.
        """
        out: List[Optional[List[float]]] = []
        for side in SIDES:
            arr_iface = self.iface_dirichlet.get(side)
            arr_ext = self.external_dirichlet.get(side)

            if arr_iface is not None and arr_ext is not None:
                raise ValueError(f"{self.name}: both interface and external Dirichlet set on {side}")

            arr = arr_iface if arr_iface is not None else arr_ext
            if arr is None:
                out.append(None)
                continue

            req = self.side_length_nodes(side)
            if len(arr) != req:
                raise ValueError(
                    f"{self.name}: Dirichlet length mismatch on {side}: got {len(arr)} expected {req}"
                )
            out.append([float(v) for v in np.asarray(arr, dtype=float)])
        return out

    def _compose_neumann_list(self) -> List[Optional[List[float]]]:
        """
        Build the Neumann list in the order expected by `heatSolver`:
        [bottom, left, top, right].

        Priority: interface Neumann (if provided) overrides external Neumann.
        """
        out: List[Optional[List[float]]] = []
        for side in SIDES:
            arr_iface = self.iface_neumann.get(side)
            arr_ext = self.external_neumann.get(side)

            if arr_iface is not None and arr_ext is not None:
                raise ValueError(f"{self.name}: both interface and external Neumann set on {side}")

            arr = arr_iface if arr_iface is not None else arr_ext
            if arr is None:
                out.append(None)
                continue

            req = self.side_length_nodes(side)
            if len(arr) != req:
                raise ValueError(
                    f"{self.name}: Neumann length mismatch on {side}: got {len(arr)} expected {req}"
                )
            out.append([float(v) for v in np.asarray(arr, dtype=float)])
        return out

    # --------------------------------------------------------------------- #
    # interface setters
    # --------------------------------------------------------------------- #
    def set_interface_dirichlet(self, side: Side, values: np.ndarray) -> None:
        """
        Set Dirichlet values along an interface side.

        Parameters
        ----------
        side : {"bottom","left","top","right"}
        values : array-like, length = node count on that side
        """
        req = self.side_length_nodes(side)
        values = np.asarray(values, dtype=float)
        if values.shape[0] != req:
            raise ValueError(f"{self.name}: Dirichlet size mismatch on {side}: {values.shape[0]} vs {req}")
        self.iface_dirichlet[side] = values

    def set_interface_neumann(self, side: Side, flux: np.ndarray) -> None:
        """
        Set Neumann flux along an interface side (positive = heat leaving the domain).

        Parameters
        ----------
        side : {"bottom","left","top","right"}
        flux : array-like, length = node count on that side
        """
        req = self.side_length_nodes(side)
        flux = np.asarray(flux, dtype=float)
        if flux.shape[0] != req:
            raise ValueError(f"{self.name}: Neumann size mismatch on {side}: {flux.shape[0]} vs {req}")
        self.iface_neumann[side] = flux

    def clear_interface_bcs(self) -> None:
        """Clear all interface boundary conditions (used at the start of each DN sub-step if needed)."""
        for s in SIDES:
            self.iface_dirichlet[s] = None
            self.iface_neumann[s] = None

    # --------------------------------------------------------------------- #
    # solve + boundary sampling
    # --------------------------------------------------------------------- #
    def solve(self, boundary_mode_return: str = "dirichlet") -> Tuple[np.ndarray, Dict[Side, np.ndarray]]:
        """
        Assemble boundary condition lists and call the single-domain solver.

        Parameters
        ----------
        boundary_mode_return : {"dirichlet","neumann"}
            Controls what boundary quantity to sample and return from the solver.

        Returns
        -------
        u_flat : (Nx*Ny,) array
            Flattened solution over the whole subdomain.
        side_values : dict[Side -> 1D array]
            Sampled boundary data in the requested mode for all four sides.
        """
        D = self._compose_dirichlet_list()
        N = self._compose_neumann_list()

        solver = heatSolver(self.dx, [self.Lx, self.Ly], D, N)
        u_flat, side_vals = solver.solve(returnDirichlet=(boundary_mode_return == "dirichlet"))

        # cache results
        self.last_u = u_flat
        if boundary_mode_return == "dirichlet":
            for i, s in enumerate(SIDES):
                self.last_boundary_dirichlet[s] = np.asarray(side_vals[i])
        else:
            for i, s in enumerate(SIDES):
                self.last_boundary_flux[s] = np.asarray(side_vals[i])

        # return as dict for convenience
        out: Dict[Side, np.ndarray] = {s: np.asarray(side_vals[i]) for i, s in enumerate(SIDES)}
        return u_flat, out

    def get_boundary_values(self, mode: str = "dirichlet") -> Dict[Side, Optional[np.ndarray]]:
        """
        Return the last cached boundary values.

        Parameters
        ----------
        mode : {"dirichlet","neumann"}

        Returns
        -------
        dict[Side -> Optional[np.ndarray]]
        """
        if mode == "dirichlet":
            return self.last_boundary_dirichlet
        if mode == "neumann":
            return self.last_boundary_flux
        raise ValueError("mode must be 'dirichlet' or 'neumann'")
