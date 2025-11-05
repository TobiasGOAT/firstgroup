from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Literal
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Side = Literal["left", "right", "bottom", "top"]

@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    width: float
    height: float

@dataclass
class BCValues:
    T_wall: float = 15.0
    T_heater: float = 40.0
    T_window: float = 5.0

class Subdomain:
    """
    Rectangular subdomain on a uniform Cartesian grid with spacing h.
    Unknowns are interior nodes only; boundary nodes are eliminated via BCs.

    NEW: Each side can accept mixed (per-node) BCs via arrays:
      - kind: array of 'D' or 'N' for Dirichlet / Neumann, length = side nodes (including corners)
      - values: array of floats, same length; for 'D' it's temperature, for 'N' it's g = ∂u/∂n

    Backward compatibility: a legacy spec with {'type': 'dirichlet'|'neumann', 'values': ...}
      is auto-expanded to per-node arrays of uniform kind.
    """

    def __init__(self, name: str, box: Box, h: float, bc_values: BCValues):
        self.name = name
        self.box = box
        self.h = float(h)
        self.bc_values = bc_values

        # 1D coordinates including boundaries
        self.x = np.arange(box.x0, box.x0 + box.width + 1e-12, self.h)
        self.y = np.arange(box.y0, box.y0 + box.height + 1e-12, self.h)
        self.Nx, self.Ny = len(self.x), len(self.y)

        # Interior unknown sizes
        self.nx_int = max(self.Nx - 2, 0)
        self.ny_int = max(self.Ny - 2, 0)
        self.N = self.nx_int * self.ny_int

        # Linear system Au = b
        self.A: Optional[sp.csr_matrix] = None
        self.b: Optional[np.ndarray] = None
        self.u_int: Optional[np.ndarray] = None

        # Per-side storage: {'kind': array[str], 'values': array[float]}
        self.boundary_data: Dict[Side, Dict[str, np.ndarray]] = {}

    # ---------- indexing helpers ----------
    def _idx(self, i: int, j: int) -> int:
        return (j - 1) * self.nx_int + (i - 1)

    def _side_len(self, side: Side) -> int:
        return self.Ny if side in ("left", "right") else self.Nx

    # ---------- assembly ----------
    def assemble_matrix(self) -> None:
        if self.nx_int <= 0 or self.ny_int <= 0:
            raise ValueError(f"{self.name}: grid too small; no interior nodes.")

        rows, cols, data = [], [], []
        self.b = np.zeros(self.N, dtype=float)

        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                k = self._idx(i, j)
                rows.append(k); cols.append(k); data.append(-4.0)
                if i - 1 >= 1:
                    rows.append(k); cols.append(self._idx(i - 1, j)); data.append(1.0)
                if i + 1 <= self.Nx - 2:
                    rows.append(k); cols.append(self._idx(i + 1, j)); data.append(1.0)
                if j - 1 >= 1:
                    rows.append(k); cols.append(self._idx(i, j - 1)); data.append(1.0)
                if j + 1 <= self.Ny - 2:
                    rows.append(k); cols.append(self._idx(i, j + 1)); data.append(1.0)

        A = sp.csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        self.A = (1.0 / self.h**2) * A

    # ---------- BC parsing ----------
    @staticmethod
    def _broadcast(arr_or_scalar, L: int) -> np.ndarray:
        if np.isscalar(arr_or_scalar):
            return np.full(L, float(arr_or_scalar), dtype=float)
        arr = np.asarray(arr_or_scalar, dtype=float).ravel()
        if arr.size != L:
            raise ValueError(f"Value length mismatch: expected {L}, got {arr.size}")
        return arr

    @staticmethod
    def _broadcast_kind(kind_or_array, L: int) -> np.ndarray:
        if isinstance(kind_or_array, (list, tuple, np.ndarray)):
            arr = np.asarray(kind_or_array)
            if arr.size != L:
                raise ValueError(f"Kind length mismatch: expected {L}, got {arr.size}")
            out = np.asarray(arr, dtype="<U1")  # 'D'/'N'
        else:
            out = np.full(L, str(kind_or_array), dtype="<U1")
        # Validate entries
        bad = ~np.isin(out, np.array(["D", "N"], dtype="<U1"))
        if np.any(bad):
            raise ValueError("Kind array must contain only 'D' or 'N'.")
        return out

    def _merge_spec(self, base_kind: np.ndarray, base_vals: np.ndarray,
                    spec: Dict, side: Side) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merge an incoming spec into (base_kind, base_vals).
        Supports legacy {'type', 'values'} or new {'kind', 'values'}.
        """
        L = self._side_len(side)
        if "kind" in spec:
            k = self._broadcast_kind(spec["kind"], L)
            v = self._broadcast(spec["values"], L)
            # Replace all (full-length)
            return k, v
        else:
            # Legacy single-type spec
            t = str(spec["type"]).lower()
            v = self._broadcast(spec["values"], L)
            if t == "dirichlet":
                k = np.full(L, "D", dtype="<U1")
            elif t == "neumann":
                k = np.full(L, "N", dtype="<U1")
            else:
                raise ValueError(f"Unknown BC type: {t}")
            return k, v

    # ---------- boundary application ----------
    def apply_boundary_conditions(
        self,
        outer_dirichlet: Dict[Side, float],
        interface_specs: Dict[str, Dict[str, object]]
    ) -> None:
        """
        Apply mixed per-node BCs:
          - Start from outer Dirichlet (uniform 'D' with given wall/heater/window).
          - Override with any interface specs, which may be legacy or mixed ('kind' + 'values').
        """
        # Always reset A,b (avoid accumulation across iterations)
        self.assemble_matrix()

        # Build per-side arrays from outer Dirichlet
        side_arrays: Dict[Side, Dict[str, np.ndarray]] = {}
        for side in ("left", "right", "bottom", "top"):
            L = self._side_len(side)
            base_kind = np.full(L, "D", dtype="<U1")
            base_vals = self._broadcast(outer_dirichlet.get(side, np.nan), L)
            side_arrays[side] = {"kind": base_kind, "values": base_vals}

        # Override with interface specs (replace-full for each side that appears)
        for _, spec in interface_specs.items():
            side: Side = spec["side"]  # type: ignore
            k_new, v_new = self._merge_spec(side_arrays[side]["kind"], side_arrays[side]["values"], spec, side)
            side_arrays[side] = {"kind": k_new, "values": v_new}

        # Save for reconstruction
        self.boundary_data = side_arrays

        # Apply to A,b per node
        A = self.A.tolil()
        h = self.h

        # vertical sides: sweep over j=1..Ny-2
        # Dirichlet: b[k] += -(1/h^2)*u_b
        # Neumann:   A[k,k] += (1/h^2) ; b[k] += -(g/h)
        # outward normals: left = -x, right = +x, bottom = -y, top = +y
        # (Signs are already encoded in g as outward derivative of THIS subdomain.)
        # Left
        kind, vals = side_arrays["left"]["kind"], side_arrays["left"]["values"]
        for j in range(1, self.Ny - 1):
            k = self._idx(1, j)
            if kind[j] == "D":
                self.b[k] += -(1.0 / h**2) * vals[j]
            else:  # 'N'
                A[k, k] += (1.0 / h**2)
                self.b[k] += -(vals[j] / h)

        # Right
        kind, vals = side_arrays["right"]["kind"], side_arrays["right"]["values"]
        for j in range(1, self.Ny - 1):
            k = self._idx(self.Nx - 2, j)
            if kind[j] == "D":
                self.b[k] += -(1.0 / h**2) * vals[j]
            else:
                A[k, k] += (1.0 / h**2)
                self.b[k] += -(vals[j] / h)

        # Bottom
        kind, vals = side_arrays["bottom"]["kind"], side_arrays["bottom"]["values"]
        for i in range(1, self.Nx - 1):
            k = self._idx(i, 1)
            if kind[i] == "D":
                self.b[k] += -(1.0 / h**2) * vals[i]
            else:
                A[k, k] += (1.0 / h**2)
                self.b[k] += -(vals[i] / h)

        # Top
        kind, vals = side_arrays["top"]["kind"], side_arrays["top"]["values"]
        for i in range(1, self.Nx - 1):
            k = self._idx(i, self.Ny - 2)
            if kind[i] == "D":
                self.b[k] += -(1.0 / h**2) * vals[i]
            else:
                A[k, k] += (1.0 / h**2)
                self.b[k] += -(vals[i] / h)

        self.A = A.tocsr()

    # ---------- solve ----------
    def solve_local(self) -> np.ndarray:
        if self.A is None or self.b is None:
            raise RuntimeError("assemble_matrix/apply_boundary_conditions must be called first.")
        self.u_int = spla.spsolve(self.A, self.b)
        return self.u_int

    # ---------- reconstruction / extraction ----------
    def field_full(self) -> np.ndarray:
        if self.u_int is None:
            raise RuntimeError("solve_local must be called before field reconstruction.")
        U = np.zeros((self.Ny, self.Nx), dtype=float)
        U[1:-1, 1:-1] = self.u_int.reshape(self.ny_int, self.nx_int)
        h = self.h

        # Left (n = -x)
        k, v = self.boundary_data["left"]["kind"], self.boundary_data["left"]["values"]
        for j in range(self.Ny):
            if k[j] == "D":
                U[j, 0] = v[j]
            else:
                U[j, 0] = U[j, 1] + h * v[j]

        # Right (n = +x)
        k, v = self.boundary_data["right"]["kind"], self.boundary_data["right"]["values"]
        for j in range(self.Ny):
            if k[j] == "D":
                U[j, -1] = v[j]
            else:
                U[j, -1] = U[j, -2] + h * v[j]

        # Bottom (n = -y)
        k, v = self.boundary_data["bottom"]["kind"], self.boundary_data["bottom"]["values"]
        for i in range(self.Nx):
            if k[i] == "D":
                U[0, i] = v[i]
            else:
                U[0, i] = U[1, i] + h * v[i]

        # Top (n = +y)
        k, v = self.boundary_data["top"]["kind"], self.boundary_data["top"]["values"]
        for i in range(self.Nx):
            if k[i] == "D":
                U[-1, i] = v[i]
            else:
                U[-1, i] = U[-2, i] + h * v[i]

        return U

    def extract_interface_values(self, side: Side) -> np.ndarray:
        U = self.field_full()
        if side == "left":   return U[:, 0].copy()
        if side == "right":  return U[:, -1].copy()
        if side == "bottom": return U[0, :].copy()
        if side == "top":    return U[-1, :].copy()
        raise ValueError(side)

    def extract_normal_derivative(self, side: Side) -> np.ndarray:
        U = self.field_full()
        h = self.h
        if side == "left":    return (U[:, 0] - U[:, 1]) / h
        if side == "right":   return (U[:, -1] - U[:, -2]) / h
        if side == "bottom":  return (U[0, :] - U[1, :]) / h
        if side == "top":     return (U[-1, :] - U[-2, :]) / h
        raise ValueError(side)
