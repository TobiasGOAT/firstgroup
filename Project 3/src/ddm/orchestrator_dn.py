# src/ddm/orchestrator_dn.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from .subdomain import RectSubdomain
from .interface import Interface

@dataclass
class DNParams:
    # Relaxation for Dirichlet traces (interface values used to drive Ω2)
    omega: float = 0.6
    # Max DN iterations and stopping tolerance on interface traces
    max_iter: int = 200
    tol: float = 1e-6

class OrchestratorDN:
    """
    Serial Dirichlet–Neumann for three subdomains:
        Ω1 --Γ1-- Ω2 --Γ2-- Ω3

    Per iteration:
      1) Solve Ω2 with Dirichlet on Γ1, Γ2 (coming from relaxed traces).
         Return Neumann (flux) on Γ1, Γ2 from Ω2.
      2) Impose those fluxes (with a minus sign) as Neumann on Ω1 and Ω3,
         then solve Ω1, Ω3 and read their Dirichlet traces on Γ1, Γ2.
      3) Relax the new Dirichlet traces and check convergence.
    """

    def __init__(self,
                 omega2: RectSubdomain,
                 omega1: RectSubdomain,
                 omega3: RectSubdomain,
                 gamma1: Interface,
                 gamma2: Interface,
                 params: DNParams):
        self.O2 = omega2
        self.O1 = omega1
        self.O3 = omega3
        self.G1 = gamma1   # interface between Ω1 (left) and Ω2 (right side)
        self.G2 = gamma2   # interface between Ω2 (left) and Ω3 (right side)
        self.p = params

        self.history: List[float] = []

        # Allocate Dirichlet trace buffers (match sliced interface lengths)
        n1 = self.O1.side_length_nodes(self.G1.left_side)   # length on Ω1 side of Γ1
        n3 = self.O3.side_length_nodes(self.G2.right_side)  # length on Ω3 side of Γ2
        self.trace_G1 = np.zeros(n1, dtype=float)
        self.trace_G2 = np.zeros(n3, dtype=float)

        # Flux under-relaxation (kept local; adjust if needed)
        self._beta = 0.1

    def iterate(self) -> Dict[str, np.ndarray]:
        """Run DN iterations and return history + converged traces."""
        # --- Initialization: produce a mild initial guess for traces
        self.O1.clear_interface_bcs()
        self.O3.clear_interface_bcs()
        self.O1.solve("dirichlet")
        self.O3.solve("dirichlet")

        # Simple constant initial guess on both interfaces
        self.trace_G1[:] = 15.0
        self.trace_G2[:] = 15.0

        for k in range(self.p.max_iter):
            # ----------------------------------------------------------
            # Step 1) Solve Ω2 with Dirichlet on Γ1, Γ2 (using traces)
            # ----------------------------------------------------------
            self.O2.clear_interface_bcs()
            # Apply Dirichlet only on the selected slices of Γ1, Γ2
            self.G1.set_dirichlet_on_right(self.trace_G1)  # Ω2(right) at Γ1
            self.G2.set_dirichlet_on_left(self.trace_G2)   # Ω2(left)  at Γ2
            self.O2.solve("neumann")  # we need flux (Neumann) returned on Γ1, Γ2

            # Read fluxes from Ω2 sides (consistent with Ω2 outward normal)
            flux_G1_from_O2 = self.G1.get_flux_from_right()
            flux_G2_from_O2 = self.G2.get_flux_from_left()

            # ----------------------------------------------------------
            # Step 2) Solve Ω1 and Ω3 with Neumann from Ω2 (minus sign)
            #         Under-relax the flux for stability
            # ----------------------------------------------------------
            beta = self._beta

            # Ω1 with Neumann on Γ1
            self.O1.clear_interface_bcs()
            self.G1.set_neumann_on_left(-beta * flux_G1_from_O2)  # minus sign!
            self.O1.solve("dirichlet")  # return Dirichlet to sample traces

            # Ω3 with Neumann on Γ2
            self.O3.clear_interface_bcs()
            self.G2.set_neumann_on_right(-beta * flux_G2_from_O2) # minus sign!
            self.O3.solve("dirichlet")

            # ----------------------------------------------------------
            # Step 3) Relax traces and check convergence
            # ----------------------------------------------------------
            new_G1 = self.G1.get_dirichlet_from_left()   # from Ω1 at Γ1 (sliced)
            new_G2 = self.G2.get_dirichlet_from_right()  # from Ω3 at Γ2 (sliced)

            old_G1 = self.trace_G1.copy()
            old_G2 = self.trace_G2.copy()

            self.trace_G1 = self.p.omega * new_G1 + (1.0 - self.p.omega) * self.trace_G1
            self.trace_G2 = self.p.omega * new_G2 + (1.0 - self.p.omega) * self.trace_G2

            err = max(np.max(np.abs(self.trace_G1 - old_G1)),
                      np.max(np.abs(self.trace_G2 - old_G2)))
            self.history.append(err)

            if err < self.p.tol:
                break

        return {
            "err_history": np.array(self.history),
            "trace_G1": self.trace_G1.copy(),
            "trace_G2": self.trace_G2.copy(),
        }
