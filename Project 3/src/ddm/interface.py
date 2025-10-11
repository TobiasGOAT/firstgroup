# src/ddm/interface.py
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .subdomain import RectSubdomain, Side

@dataclass
class Interface:
    """
    Interface between two rectangular subdomains with optional index slicing
    on each side to match different side lengths (e.g., tall Ω2 vs unit Ω1/Ω3).

    left_side/right_side must be one of: "bottom", "left", "top", "right".
    Spans are half-open [start, stop) in node indices along the side.
    """
    left: RectSubdomain
    right: RectSubdomain
    left_side: Side
    right_side: Side
    left_span: Optional[Tuple[int, int]] = None   # [start, stop)
    right_span: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        # Validate and materialize slices
        nL = self.left.side_length_nodes(self.left_side)
        nR = self.right.side_length_nodes(self.right_side)

        L0, L1 = (0, nL) if self.left_span  is None else self.left_span
        R0, R1 = (0, nR) if self.right_span is None else self.right_span

        if not (0 <= L0 < L1 <= nL and 0 <= R0 < R1 <= nR):
            raise ValueError("Invalid interface spans")
        if (L1 - L0) != (R1 - R0):
            raise ValueError(
                f"Interface node count mismatch after slicing: {(L1-L0)} vs {(R1-R0)}"
            )

        self._Lslice = slice(L0, L1)
        self._Rslice = slice(R0, R1)

    # ---------------------------
    # Helpers
    # ---------------------------
    def _background_dirichlet(self, sd: RectSubdomain, side: str) -> float:
        """
        Pick a reasonable background Dirichlet value for the non-interface
        portion of a side. If an external Dirichlet is available on that
        side, use its (finite) median; otherwise fall back to 15.0 (ambient).
        """
        try:
            ext = sd.external_dirichlet.get(side, None)
        except Exception:
            ext = None

        if ext is None:
            return 15.0

        # ext can be an array or scalar
        try:
            arr = np.asarray(ext, dtype=float)
            if arr.ndim == 0:
                return float(arr)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return 15.0
            return float(np.median(finite))
        except Exception:
            # As a last resort
            try:
                return float(ext)
            except Exception:
                return 15.0

    # ---------------------------
    # Read cached values (sliced)
    # ---------------------------
    def get_dirichlet_from_left(self) -> np.ndarray:
        arr = self.left.get_boundary_values("dirichlet")[self.left_side]
        if arr is None:
            raise RuntimeError("Left side has no cached Dirichlet")
        return np.asarray(arr, float)[self._Lslice]

    def get_dirichlet_from_right(self) -> np.ndarray:
        arr = self.right.get_boundary_values("dirichlet")[self.right_side]
        if arr is None:
            raise RuntimeError("Right side has no cached Dirichlet")
        return np.asarray(arr, float)[self._Rslice]

    def get_flux_from_left(self) -> np.ndarray:
        arr = self.left.get_boundary_values("neumann")[self.left_side]
        if arr is None:
            raise RuntimeError("Left side has no cached Neumann")
        return np.asarray(arr, float)[self._Lslice]

    def get_flux_from_right(self) -> np.ndarray:
        arr = self.right.get_boundary_values("neumann")[self.right_side]
        if arr is None:
            raise RuntimeError("Right side has no cached Neumann")
        return np.asarray(arr, float)[self._Rslice]

    # ---------------------------
    # Impose BC on slices only
    # Build full-length arrays so solver sees consistent shapes
    # ---------------------------
    def set_dirichlet_on_left(self, values: np.ndarray):
        nL = self.left.side_length_nodes(self.left_side)
        full = np.full(nL, self._background_dirichlet(self.left, self.left_side), dtype=float)
        full[self._Lslice] = np.asarray(values, float)
        self.left.set_interface_dirichlet(self.left_side, full)

    def set_dirichlet_on_right(self, values: np.ndarray):
        nR = self.right.side_length_nodes(self.right_side)
        full = np.full(nR, self._background_dirichlet(self.right, self.right_side), dtype=float)
        full[self._Rslice] = np.asarray(values, float)
        self.right.set_interface_dirichlet(self.right_side, full)

    def set_neumann_on_left(self, flux: np.ndarray):
        nL = self.left.side_length_nodes(self.left_side)
        full = np.zeros(nL, dtype=float)  # zero flux on non-interface portion
        full[self._Lslice] = np.asarray(flux, float)
        self.left.set_interface_neumann(self.left_side, full)

    def set_neumann_on_right(self, flux: np.ndarray):
        nR = self.right.side_length_nodes(self.right_side)
        full = np.zeros(nR, dtype=float)  # zero flux on non-interface portion
        full[self._Rslice] = np.asarray(flux, float)
        self.right.set_interface_neumann(self.right_side, full)
